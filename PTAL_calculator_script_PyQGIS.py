from qgis.core import (QgsProcessing,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFile,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterField,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterFeatureSink,
                       QgsProcessingException,
                       QgsCoordinateReferenceSystem,
                       QgsCoordinateTransform,
                       QgsFeature,
                       QgsFeatureSink,
                       QgsField,
                       QgsFields,
                       QgsWkbTypes)
from qgis.PyQt.QtCore import QVariant
import numpy as np
import pandas as pd

CRS_METERS = "EPSG:28992"       # distances are calculated in RD New (meters)
ROUTE_TYPES = [0, 1, 2, 3, 4]   # tram, subway, rail, bus, ferry
SPEEDS = [70, 300]              # m/min: walking, cycling
BUFFERS = [700, 3000]         # m: distance covered in 1 hour (walking, cycling)


class PTALAccessIndexV5(QgsProcessingAlgorithm):
    DATASET = "DATASET"
    STOPS = "STOPS"
    STOP_ID = "STOP_ID"
    CENTROIDS = "CENTROIDS"
    CENTROID_ID = "CENTROID_ID"
    ROUTE = "ROUTE"
    TRAVEL = "TRAVEL"
    OUTPUT = "OUTPUT"

    def name(self):
        return "ptal_access_index_v5"

    def displayName(self):
        return "PTAL Access Index (v5)"

    def group(self):
        return "PTAL"

    def groupId(self):
        return "ptal"

    def shortHelpString(self):
        return ("Calculates the distance matrix between centroids and stops "
                "(straight-line distance in meters) and uses it to compute the "
                "Access Index (AI) per centroid.\n"
                "AI = largest EDF + 0.5 * sum of all other EDFs.\n"
                "Walking: 70 m/min, buffer 700 m. Cycling: 300 m/min, buffer 3000 m.\n"
                "Stops outside the buffer are ignored (NaN).\n"
                "The result is a new point layer with centroid_id and AI.\n"
                "Centroids without a stop inside the buffer get AI = 0.\n"
                "The dataset must already be filtered on day and time window "
                "(e.g. big_dataset_morningpeak_13okt.csv).")

    def createInstance(self):
        return PTALAccessIndexV5()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFile(
            self.DATASET, "Dataset (e.g. big_dataset_morningpeak_13okt.csv)",
            extension="csv"))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.STOPS, "Stops layer (points)",
            [QgsProcessing.TypeVectorPoint]))
        self.addParameter(QgsProcessingParameterField(
            self.STOP_ID, "Field with station_id (e.g. stoparea:17807)",
            parentLayerParameterName=self.STOPS,
            type=QgsProcessingParameterField.Any))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.CENTROIDS, "Centroids layer (points)",
            [QgsProcessing.TypeVectorPoint]))
        self.addParameter(QgsProcessingParameterField(
            self.CENTROID_ID, "Field with centroid ID",
            parentLayerParameterName=self.CENTROIDS,
            type=QgsProcessingParameterField.Any))
        self.addParameter(QgsProcessingParameterEnum(
            self.ROUTE, "Mode (route_type)",
            options=["Tram (0)", "Subway (1)", "Rail (2)", "Bus (3)", "Ferry (4)"],
            defaultValue=2))
        self.addParameter(QgsProcessingParameterEnum(
            self.TRAVEL, "Travel type to the stop",
            options=["Walking (70 m/min, buffer 4200 m)",
                     "Cycling (300 m/min, buffer 18000 m)"],
            defaultValue=0))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, "Centroids with AI", QgsProcessing.TypeVectorPoint))

    # ---------- Helper: read points from a layer ----------
    def _points(self, source, id_field, target_crs, context):
        transform = None
        if source.sourceCrs() != target_crs:
            transform = QgsCoordinateTransform(source.sourceCrs(), target_crs,
                                               context.transformContext())
        ids, xs, ys = [], [], []
        for f in source.getFeatures():
            geom = f.geometry()
            if geom is None or geom.isNull() or geom.isEmpty():
                continue
            p = geom.centroid().asPoint()
            if transform is not None:
                p = transform.transform(p)
            ids.append(str(f[id_field]))
            xs.append(p.x())
            ys.append(p.y())
        return pd.DataFrame({"id": ids, "x": xs, "y": ys})

    def processAlgorithm(self, parameters, context, feedback):
        dataset_path = self.parameterAsFile(parameters, self.DATASET, context)
        stops_src = self.parameterAsSource(parameters, self.STOPS, context)
        stop_id = self.parameterAsString(parameters, self.STOP_ID, context)
        cent_src = self.parameterAsSource(parameters, self.CENTROIDS, context)
        cent_id = self.parameterAsString(parameters, self.CENTROID_ID, context)
        route_type = ROUTE_TYPES[self.parameterAsEnum(parameters, self.ROUTE, context)]
        travel_idx = self.parameterAsEnum(parameters, self.TRAVEL, context)
        speed = SPEEDS[travel_idx]
        buffer_m = BUFFERS[travel_idx]

        feedback.pushInfo("route_type {}, buffer {} m, speed {} m/min".format(
            route_type, buffer_m, speed))

        # ---------- Filter dataset on mode ----------
        feedback.pushInfo("Reading dataset...")
        dataset = pd.read_csv(dataset_path)

        # Some versions of the dataset have x/y instead of stop_lon/stop_lat
        if "stop_lat" not in dataset.columns and "y" in dataset.columns:
            dataset = dataset.rename(columns={"x": "stop_lon", "y": "stop_lat"})

        dataset = dataset[dataset["route_type"] == route_type]
        if dataset.empty:
            raise QgsProcessingException(
                "No trips found for this mode (route_type {}).".format(route_type))

        small = dataset[["stop_name", "station_id", "stop_lat", "stop_lon",
                         "trip_id", "departure_time"]]
        small = small.drop_duplicates(subset=["station_id", "trip_id"])

        stop_info = (small.groupby("station_id")
                     .agg(stop_name=("stop_name", "first"),
                          frequency=("trip_id", lambda s: s.nunique() / 2.5),
                          stop_lat=("stop_lat", "mean"),
                          stop_lon=("stop_lon", "mean"))
                     .reset_index())
        stop_info["SWT"] = 0.5 * (60 / stop_info["frequency"])
        stop_info["AWT"] = stop_info["SWT"] + 0.75
        feedback.pushInfo("{} stations".format(len(stop_info)))

        # ---------- Build the distance matrix ----------
        feedback.pushInfo("Calculating distance matrix...")
        target_crs = QgsCoordinateReferenceSystem(CRS_METERS)

        stops = self._points(stops_src, stop_id, target_crs, context)
        stops = stops.drop_duplicates(subset="id")
        # only stations that occur in the dataset for this mode
        stops = stops[stops["id"].isin(stop_info["station_id"])]
        if stops.empty:
            raise QgsProcessingException(
                "None of the points in the stops layer has a station_id that "
                "occurs in the dataset. Check the selected ID field.")

        cents = self._points(cent_src, cent_id, target_crs, context)
        cents = cents.drop_duplicates(subset="id")
        feedback.pushInfo("{} centroids x {} stations".format(len(cents), len(stops)))

        dx = cents["x"].to_numpy()[:, None] - stops["x"].to_numpy()[None, :]
        dy = cents["y"].to_numpy()[:, None] - stops["y"].to_numpy()[None, :]
        distances = pd.DataFrame(np.hypot(dx, dy),
                                 index=cents["id"].to_numpy(),
                                 columns=stops["id"].to_numpy())
        distances.index.name = "ID"

        # Distances above the buffer become NaN (not 0!)
        distances = distances.where(distances <= buffer_m, np.nan)

        times = distances / speed
        times = times.T
        times.index.name = "station_id"

        station_times = stop_info.merge(times, on="station_id")
        if station_times.empty:
            raise QgsProcessingException("No stations left after merging.")

        # ---------- Wide to long ----------
        attr = ["station_id", "stop_name", "frequency", "stop_lat", "stop_lon",
                "SWT", "AWT"]
        centroid_cols = [c for c in station_times.columns if c not in attr + ["id"]]

        melted = station_times.melt(id_vars=attr,
                                    value_vars=centroid_cols,
                                    var_name="centroid_id",
                                    value_name="travel_times")
        melted = melted.dropna(subset=["travel_times"])
        melted = melted[["centroid_id"] + attr + ["travel_times"]]
        melted = melted.sort_values("centroid_id").reset_index(drop=True)
        melted["travel_times"] = melted["travel_times"].astype(float)

        # ---------- TAT, EDF and AI ----------
        if melted.empty:
            feedback.pushInfo("No centroid lies within the buffer of a station: "
                              "all centroids get AI = 0.")
            result = pd.DataFrame({"centroid_id": [], "AI": []})
        else:
            melted["TAT"] = melted["travel_times"] + melted["AWT"]
            melted["EDF"] = 0.5 * (60 / melted["TAT"])

            cgroup = melted.groupby(by="centroid_id")["EDF"]
            max_edf = cgroup.transform("max")
            sum_edf = cgroup.transform("sum")
            melted["AI"] = max_edf + 0.5 * (sum_edf - max_edf)

            result = (melted[["centroid_id", "AI"]]
                      .drop_duplicates("centroid_id")
                      .reset_index(drop=True))

        # ---------- Create the new layer ----------
        ai_dict = dict(zip(result["centroid_id"].astype(str), result["AI"]))

        fields = QgsFields()
        fields.append(QgsField("centroid_id", QVariant.String))
        fields.append(QgsField("AI", QVariant.Double))

        sink, dest_id = self.parameterAsSink(
            parameters, self.OUTPUT, context, fields,
            QgsWkbTypes.Point, cent_src.sourceCrs())
        if sink is None:
            raise QgsProcessingException("Could not create the output layer.")

        seen = set()
        n_zero = 0
        for f in cent_src.getFeatures():
            cid = str(f[cent_id])
            if cid in seen:
                continue
            geom = f.geometry()
            if geom is None or geom.isNull() or geom.isEmpty():
                continue
            seen.add(cid)
            if cid in ai_dict:
                ai = float(ai_dict[cid])
            else:
                ai = 0.0            # no stop within the buffer
                n_zero += 1
            feat = QgsFeature(fields)
            feat.setGeometry(geom.centroid())
            feat.setAttributes([cid, ai])
            sink.addFeature(feat, QgsFeatureSink.FastInsert)

        feedback.pushInfo("{} centroids in the new layer ({} with AI = 0)".format(
            len(seen), n_zero))

        return {self.OUTPUT: dest_id}
