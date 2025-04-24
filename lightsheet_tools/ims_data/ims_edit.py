from pathlib import Path
import numpy as np
from collections.abc import Sequence, MutableSequence
import h5py
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, ElementTree
from dataclasses import dataclass
from copy import deepcopy
from io import BytesIO


@dataclass
class Color:
    """https://doc.babylonjs.com/features/featuresDeepDive/materials/using/materials_introduction"""

    ambient_color: tuple[float, float, float] = 0., 0., 0.

    diffuse_color: tuple[float, float, float] = 0., 0., 0.

    specular_color: tuple[float, float, float] = 0., 0., 0.

    emissive_color: tuple[float, float, float] = 0., 0., 0.

    shininess: float = 0.

    transparency: float = 0.

    def get_xml_str(self, name: str) -> str:
        match name:
            case "ambient" | "diffuse" | "specular" | "emissive":
                value = getattr(self, f"{name}_color")
                value = [min(1, max(0, v)) for v in value]
                value = [f"{float(v):0.3f}" for v in value]
                value = ",".join(value)
                return f"(({value}))"
            case "shininess" | "transparency":
                value = getattr(self, name)
                return f"({float(value):0.3f})"
            case _:
                assert False


class Spots:

    name: str

    points: np.ndarray

    n_points: int

    radius: np.ndarray

    color: Color

    def __init__(
            self, name: str, points: np.ndarray | Sequence[tuple[float, float, float]],
            radius: np.ndarray | Sequence[float] | float, color: Color | None = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.name = name
        self.points = np.asarray(points)
        self.n_points = len(self.points)
        if color is None:
            color = Color(diffuse_color=(.8, 0., .2))
        self.color = color

        if isinstance(radius, (np.ndarray, Sequence)):
            self.radius = np.asarray(radius)
        else:
            self.radius = np.ones(self.n_points) * radius

    def __str__(self):
        return f"<{self.name} ({self.n_points} cells) {id(self)}@{self.__class__.__qualname__}>"

    def __repr__(self):
        return self.__str__()


class ImsFile:

    ims_filename: Path

    xml_filename = Path(__file__).parent / "ims_metadata.xml"

    spots: list[Spots]

    _xml_tree: ElementTree

    _points_container: Element

    _template_point: Element

    def __init__(self, ims_filename: Path, **kwargs):
        super().__init__(**kwargs)
        self.ims_filename = ims_filename
        self.spots = []

        self._xml_tree = ET.parse(self.xml_filename)
        root = self._xml_tree.getroot()
        self._points_container = root.find("bpSurfaceApplication").find("bpComponentGroup").find("bpSurfaceComponent")
        self._template_point = deepcopy(self._points_container.find("bpPointsViewer"))

        with h5py.File(str(self.ims_filename), "r") as file:
            if "Scene8" in file and "Data" in file["Scene8"]:
                buffer = BytesIO()
                buffer.write(file["Scene8"]["Data"][0].tobytes())
                buffer.seek(0)

                self._xml_tree = ET.parse(buffer)
                root = self._xml_tree.getroot()
                self._points_container = root.find(
                    "bpSurfaceApplication").find("bpComponentGroup").find("bpSurfaceComponent")

        for item in list(self._points_container.findall("bpPointsViewer")):
            self._points_container.remove(item)

        self._read_spots_from_file()

    def _read_spots_from_file(self) -> None:
        with h5py.File(str(self.ims_filename), "r") as file:
            if "Scene8" not in file or "Content" not in file["Scene8"]:
                return

            scene_content = file["Scene8"]["Content"]
            for i in range(scene_content.attrs["NumberOfPoints"]):
                points = scene_content[f"Points{i}"]

                data = np.asarray(points["Spot"])
                pos = np.empty((len(data), 3), dtype=np.float32)
                radius = np.empty((len(data), ), dtype=np.float32)

                pos[:, 0] = data["PositionX"]
                pos[:, 1] = data["PositionY"]
                pos[:, 2] = data["PositionZ"]
                radius[:] = data["Radius"]

                spots = Spots(
                    name=points.attrs["Name"].tobytes().decode(),
                    points=pos,
                    radius=radius,
                )
                self.spots.append(spots)

    def __len__(self):
        return len(self.spots)

    def __getitem__(self, index):
        return self.spots[index]

    def __setitem__(self, index, spots):
        if isinstance(index, slice):
            for spot in spots:
                self._assert_is_spots(spot)
        else:
            self._assert_is_spots(spots)
        self.spots[index] = spots

    def __delitem__(self, index):
        del self.spots[index]

    def append(self, spots: Spots):
        self._assert_is_spots(spots)
        self.spots.append(spots)

    def extend(self, spots: list[Spots]):
        for spot in spots:
            self._assert_is_spots(spot)
            self.spots.append(spot)

    def insert(self, index, spots: Spots):
        self._assert_is_spots(spots)
        self.spots.insert(index, spots)

    def remove(self, spots: Spots):
        self._assert_is_spots(spots)
        self.spots.remove(spots)

    def _assert_is_spots(self, e):
        if not isinstance(e, Spots):
            raise TypeError(f'Expected a Spots, not {type(e).__name__}')

    def write_content(self):
        with h5py.File(str(self.ims_filename), "a") as file:
            if "Scene8" in file and "Content" in file["Scene8"]:
                scene_content = file["Scene8"]["Content"]
                if "NumberOfPoints" in scene_content.attrs:
                    n = scene_content.attrs["NumberOfPoints"]
                    for i in range(n):
                        del scene_content[f"Points{i}"]
                    scene_content.attrs["NumberOfPoints"] = np.uint64(0)

            for i, spot in enumerate(self.spots):
                self._add_scene8(file, spot, i, len(self.spots))

            if "Scene8" in file and "Data" in file["Scene8"]:
                del file["Scene8"]["Data"]
            self._add_xml_data(file)

    def _add_scene(self, file: h5py.File, name: str, points: np.ndarray, radius: np.ndarray):
        if len(points) != len(radius):
            raise ValueError("Points and radius must have same number of elements")
        n = len(points)

        if "Scene" in file:
            scene_content = file["Scene"]["Content"]
            i = scene_content.attrs["NumberOfPoints"]
            scene_content.attrs["NumberOfPoints"] = i + 1
            points_name = f"Points{i}"
        else:
            scene = file.create_group("/Scene")
            scene_content = scene.create_group("Content")
            scene_content.attrs["NumberOfPoints"] = np.uint64(1)
            points_name = "Points0"

        sc_points = scene_content.create_group(points_name)

        attrs = {
            "Description": b"",
            "HistoryLog": b"",
            "Material":
                b'<bpMaterial mDiffusion="0.8 0.8 0.8" mSpecular="0.2 0.2 0.2" mEmission="0 0 0" mAmbient="0 0 0" '
                b'mTransparency="0" mShinyness="0.1"/>\n',
            "Name": name.encode(),
            "Unit": b"um",
        }
        for key, value in attrs.items():
            sc_points.attrs[key] = np.array([value])
        sc_points.attrs["Id"] = np.uint64(200000 + scene_content.attrs["NumberOfPoints"])

        ts = b"".join(file["DataSetInfo"]["TimeInfo"].attrs["TimePoint1"])
        sc_points.create_dataset("TimeInfos", data=np.array([ts]))
        sc_points.create_dataset("Time", data=np.zeros((n, 1), dtype=np.int64), maxshape=(None, 1))
        dset = sc_points.create_dataset("CoordsXYZR", (n, 4), maxshape=(None, 4), dtype=np.float32)
        dset[:, :3] = points
        dset[:, 3] = radius

    def _add_scene8(self, file: h5py.File, spots: Spots, spot_num: int, total_spots: int):
        n_points = spots.n_points
        if "Scene8" in file:
            scene = file["Scene8"]
            if "Content" in scene:
                scene_content = scene["Content"]
            else:
                scene_content = scene.create_group("Content")
        else:
            scene = file.create_group("Scene8")
            scene_content = scene.create_group("Content")

        scene_content.attrs["NumberOfPoints"] = np.uint64(total_spots)
        points_name = f"Points{spot_num}"

        sc_points = scene_content.create_group(points_name)

        attrs = {
            "CreationParameters":
                b'<bpPointsCreationParameters mFormatVersion="9.5" mEnableRegionOfInterest="false" '
                b'mEnableRegionGrowing="false" mEnableTracking="false" mProcessEntireImage="false" '
                b'mSourceImageIndex="0" mSourceChannelIndex="0" mEstimatedDiameter="9.944 9.944 9.944" '
                b'mBackgroundSubtraction="true" mRegionGrowingType="eLocalContrast" '
                b'mRegionGrowingAutomaticTreshold="true" mRegionGrowingManualThreshold="0" '
                b'mRegionGrowingDiameter="eDiameterFromVolume" mCreateRegionChannel="false" '
                b'mEnableShortestDistance="true">\n<bpRegionOfInterestContainer>\n</bpRegionOfInterestContainer>'
                b'\n<mSpotFilter>\n<bpStatisticsFilter mLowerThresholdEnable="true" mLowerThresholdManual="false" '
                b'mLowerThresholdManualInitToAuto="false" mLowerThresholdManualValue="36.1219" '
                b'mUpperThresholdEnable="false" mUpperThresholdManual="false" mUpperThresholdManualInitToAuto="false" '
                b'mUpperThresholdManualValue="1" mSelectHigh="true" mManualThreshold="false" '
                b'mManualThresholdValue="36.122" mInitManualThresholdToAuto="false">\n  <bpStatisticsValueType '
                b'mName="Quality" mUnit="" mFactors="0"/>\n</bpStatisticsFilter>\n</mSpotFilter>\n'
                b'<bpObjectTrackingAlgorithmParameters TrackAlgoName="Autoregressive Motion" mFillGapEnable="false" '
                b'mReferenceFramesId="0">\n<ObjectTrackingAlgorithmLinearAsignment MaxGapSize="3" '
                b'MaxDistance="-1"/>\n</bpObjectTrackingAlgorithmParameters>\n<mTrackFilter>\n<bpStatisticsFilter '
                b'mLowerThresholdEnable="true" mLowerThresholdManual="true" mLowerThresholdManualInitToAuto="false" '
                b'mLowerThresholdManualValue="2.5" mUpperThresholdEnable="false" mUpperThresholdManual="false" '
                b'mUpperThresholdManualInitToAuto="true" mUpperThresholdManualValue="1" mSelectHigh="true" '
                b'mManualThreshold="true" mManualThresholdValue="2.500" mInitManualThresholdToAuto="false">\n  '
                b'<bpStatisticsValueType mName="Track Duration Steps" mUnit="" mFactors="0"/>\n</bpStatisticsFilter>\n'
                b'</mTrackFilter>\n</bpPointsCreationParameters>\n',
            "CreatorName": b"Surpass",
            "Description": b"",
            "HistoryLog": b"",
            "Material":
                b'<bpMaterial mDiffusion="0.8 0.8 0.8" mSpecular="0.2 0.2 0.2" mEmission="0 0 0" mAmbient="0 0 0" '
                b'mTransparency="0" mShinyness="0.1"/>\n',
            "Name": spots.name.encode(),
            "ObjectGUID": b"",
            "Unit": b"um",
        }
        for key, value in attrs.items():
            sc_points.attrs[key] = np.array([value])
        sc_points.attrs["Id"] = np.uint64(200000 + spot_num + 1)
        sc_points.attrs["SpotZStretchFactor"] = np.float32(1)
        sc_points.attrs["TrackSegment0_FocusEnabled"] = np.uint64(1)

        sc_points.create_dataset(
            "Category",
            data=np.array([(0, b'Spot', b'Spot'), (1, b'Overall', b'Overall')],
                          dtype=[('ID', '<i8'), ('CategoryName', 'S256'), ('Name', 'S256')])
        )
        sc_points.create_dataset(
            "Factor",
            data=np.array([(1, b'Collection', b'Diameter'),
                           (2, b'Collection', b'Distance from Origin'),
                           (3, b'Image', b'Image 1'), (4, b'Channel', b'1'),
                           (4, b'Image', b'Image 1'), (5, b'Collection', b'Position'),
                           (6, b'Spots', spots.name.encode())],
                          dtype=[('ID_List', '<i8'), ('Name', 'S256'), ('Level', 'S256')])
        )
        sc_points.create_dataset(
            "FactorList",
            data=np.array([(1,), (2,), (3,), (4,), (5,), (6,)], dtype=[('ID', '<i8')])
        )
        sc_points.create_dataset(
            "LabelGroupNames",
            data=np.array([], dtype=[('LabelGroupName', 'S256'), ('EndLabelValue', '<i8')])
        )
        sc_points.create_dataset(
            "LabelSetLabelIDs",
            data=np.array([], dtype=[('IDLabel', '<i8')])
        )
        sc_points.create_dataset(
            "LabelSetObjectIDs",
            data=np.array([], dtype=[('IDObject', '<i8')])
        )
        sc_points.create_dataset(
            "LabelSets",
            data=np.array([], dtype=[('EndLabelIDs', '<i8'), ('EndObjectIDs', '<i8')])
        )
        sc_points.create_dataset(
            "LabelValues",
            data=np.array([], dtype=[('LabelValue', 'S256')])
        )
        sc_points.create_dataset(
            "MainTrackSegmentTable",
            data=np.array([(b'0', b'TrackSegment0')],
                          dtype=[('ObjectsName', 'S256'), ('TrackSegmentName', 'S256')])
        )
        sc_points.create_dataset(
            "MainTrackSegmentTable_Focus",
            data=np.array([(b'0', b'TrackSegment0_Focus')],
                          dtype=[('ObjectsName', 'S256'), ('TrackSegmentName', 'S256')])
        )
        sc_points.create_dataset(
            "StatisticsType",
            data=np.array([(35, 0, 0, b'Area', b'um^2'), (36, 0, 1, b'Diameter X', b'um'),
                           (37, 0, 1, b'Diameter Y', b'um'), (38, 0, 1, b'Diameter Z', b'um'),
                           (39, 0, 2, b'Distance from Origin', b'um'),
                           (40, 0, 3, b'Distance to Image Border XY', b'um'),
                           (41, 0, 3, b'Distance to Image Border XYZ', b'um'),
                           (42, 0, 4, b'Intensity Center', b''),
                           (43, 0, 4, b'Intensity Max', b''),
                           (44, 0, 4, b'Intensity Mean', b''),
                           (45, 0, 4, b'Intensity Median', b''),
                           (46, 0, 4, b'Intensity Min', b''),
                           (47, 0, 4, b'Intensity StdDev', b''),
                           (48, 0, 4, b'Intensity Sum', b''),
                           (49, 0, 4, b'Intensity Sum of Square', b''),
                           (50, 1, 0, b'Number of Spots per Time Point', b''),
                           (51, 0, 3, b'Number of Voxels', b''),
                           (52, 0, 5, b'Position X', b'um'), (53, 0, 5, b'Position Y', b'um'),
                           (54, 0, 5, b'Position Z', b'um'),
                           (55, 0, 6, b'Shortest Distance to Spots', b''),
                           (56, 0, 0, b'Time', b's'), (57, 0, 0, b'Time Index', b''),
                           (9, 1, 0, b'Total Number of Spots', b''),
                           (58, 0, 0, b'Volume', b'um^3')],
                          dtype=[('ID', '<i8'), ('ID_Category', '<i8'), ('ID_FactorList', '<i8'), ('Name', 'S256'),
                                 ('Unit', 'S256')])
        )
        sc_points.create_dataset(
            "StatisticsValueTimeOffset",
            data=np.array([(-1, 0, 1), (0, 1, 1)],
                          dtype=[('ID', '<i8'), ('IndexBegin', '<i8'), ('IndexEnd', '<i8')])
        )
        sc_points.create_dataset(
            "Time",
            data=np.array([(0, 0, 1000000000, 0)],
                          dtype=[('ID', '<i8'), ('Birth', '<i8'), ('Death', '<i8'), ('IDTimeBegin', '<i8')])
        )

        ts = b"".join(file["DataSetInfo"]["TimeInfo"].attrs["TimePoint1"])
        sc_points.create_dataset(
            "TimeBegin",
            data=np.array([(0, ts)],
                          dtype=[('ID', '<i8'), ('ObjectTimeBegin', 'S256')])
        )

        sc_points.create_dataset(
            "StatisticsValue",
            data=np.array([(-1, -1, 9, n_points)],
                          dtype=[('ID_Time', '<i8'), ('ID_Object', '<i8'), ('ID_StatisticsType', '<i8'), ('Value', '<f8')])
        )
        sc_points.create_dataset(
            "SpotTimeOffset",
            data=np.array([(0, 0, n_points)],
                          dtype=[('ID', '<i8'), ('IndexBegin', '<i8'), ('IndexEnd', '<i8')]),
            maxshape=(None,),
        )
        sc_points.create_dataset(
            "Spot",
            data=np.array([(i, *p, r) for i, p, r in zip(np.arange(n_points), spots.points, spots.radius)],
                          dtype=[('ID', '<i8'), ('PositionX', '<f4'), ('PositionY', '<f4'), ('PositionZ', '<f4'),
                                 ('Radius', '<f4')]),
            maxshape=(None,),
        )
        sc_points.create_dataset(
            "TrackSegment0",
            data=np.array([(i, 0, i) for i in range(n_points)],
                          dtype=[('ID', '<i8'), ('FirstObjectId', '<i8'), ('Position', '<f8')]),
            maxshape=(None,),
        )
        sc_points.create_dataset(
            "TrackSegment0_Focus",
            data=np.array([(i, 0, i) for i in range(n_points)],
                          dtype=[('ID', '<i8'), ('FirstObjectId', '<i8'), ('Position', '<f8')]),
            maxshape=(None,),
        )

    def _add_xml_data(self, file: h5py.File):
        for i, spot in enumerate(self.spots):
            points = deepcopy(self._template_point)
            points.find("bpSurfaceComponent").find("name").text = spot.name

            material = points.find("bpSurfaceComponent").find("material")
            material.find("ambientColor").text = spot.color.get_xml_str("ambient")
            material.find("diffuseColor").text = spot.color.get_xml_str("diffuse")
            material.find("specularColor").text = spot.color.get_xml_str("specular")
            material.find("emissiveColor").text = spot.color.get_xml_str("emissive")
            material.find("shininess").text = spot.color.get_xml_str("shininess")
            material.find("transparency").text = spot.color.get_xml_str("transparency")

            points.find("bpPointsId").set("Value", str(200000 + i + 1))
            self._points_container.append(points)

        buffer = BytesIO()
        self._xml_tree.write(buffer, encoding="utf-8", xml_declaration=True)

        if "Scene8" in file:
            scene = file["Scene8"]
            if "Data" in scene:
                data = scene["Data"]
            else:
                data = scene.create_dataset("Data", (1,), dtype=h5py.vlen_dtype(np.dtype('int8')))
        else:
            scene = file.create_group("Scene8")
            data = scene.create_dataset("Data", (1,), dtype=h5py.vlen_dtype(np.dtype('int8')))

        data[0] = np.frombuffer(buffer.getvalue(), dtype=np.int8)


if __name__ == "__main__":
    ims = ImsFile(
        ims_filename=Path(r"C:\Users\CPLab\Downloads\ims\Microglia-vasculature_with_objects2_C1_Z000_spots2x.ims")
    )
    ims.append(Spots(
        name="Cheese",
        points=[(100, 100, 100), (100, 2100, 100), (100, 200, 100)],
        radius=10,
        color=Color(diffuse_color=(0, 1, 0)),
    ))
    ims.append(Spots(
        name="fruit",
        points=[(100, 100, 100), (100, 2100, 100), (100, 200, 100)],
        radius=22,
        color=Color(diffuse_color=(0, 0, 1)),
    ))

    print(ims.spots)
    ims.write_content()
