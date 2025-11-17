from functools import wraps
import multiprocessing as mp
import tempfile
import platform
from typing import IO
from collections.abc import Sequence
import datetime
import glob
from tqdm import tqdm
import logging
import shutil
from itertools import cycle
import math
import subprocess
import hashlib
import time
import psutil
from contextlib import contextmanager
import re
import tifffile
from functools import partial
import imagej
import scyjava
from pathlib import Path
from ome_types import from_tiff
from xsdata.formats.dataclass.parsers.config import ParserConfig

dim_order = {0: 'x', 1: 'y', 2: 'z', 3: 't', 4: 'c'}

wavelength_color = {
    488: "4AFF00",
    561: "DEFF00",
    640: "FF0000",
}


def now() -> str:
    return datetime.datetime.now().isoformat()


def measure_memory(sleep_duration: float = 2 * 60, interval: float = 1):
    max_mem = 0
    psutil.cpu_percent(interval)  # first call is bad data
    time.sleep(max(sleep_duration - interval, 0))
    max_cpu = psutil.cpu_percent(interval)

    while True:
        virtual_mem = psutil.virtual_memory()
        used = virtual_mem.total - virtual_mem.available
        max_mem = max(used, max_mem)

        cpu = psutil.cpu_percent(interval)
        max_cpu = max(cpu, max_cpu)

        print(f"Used: {round(used / 1028 ** 3)}GB \t Max: {round(max_mem / 1028 ** 3)}GB. CPU: {cpu}% \t Max CPU: {max_cpu}%")

        time.sleep(max(sleep_duration - interval, 0))


def convert_bytes(num):
    step_unit = 1024

    for x in ['bytes', 'KB', 'MB', 'GB']:
        if num < step_unit:
            return f"{num:3.1f} {x}"
        num /= step_unit

    return f"{num:.1f} TB"


def get_drive_name(path: Path | str):
    drive = Path(path).drive
    if platform.system() != "Windows":
        return drive
    try:
        val = subprocess.check_output(["cmd", f"/c vol {drive}"]).decode().splitlines()[0].split(" ")[-1]
        if val == "label.":
            val = drive
    except Exception:
        val = drive
    return val


class ProcAttr:
    obj_callee = None

    def __init__(self, obj_callee):
        self.obj_callee = obj_callee

    def __getattr__(self, item):
        if hasattr(self.obj_callee, item):
            original_func = getattr(self.obj_callee, item)

            @wraps(original_func)
            def callback(*args, **kwargs):
                return self.call_in_subprocess(original_func, args, kwargs)

            return callback

        raise AttributeError(f"Cannot find '{item}' in {self}")

    def call_in_subprocess(self, f, args, kwargs):
        queue = mp.Queue()
        p = mp.Process(target=ProcAttr._run_func, name="IJ process", args=(f, args, kwargs, queue))
        p.start()

        msg, value = queue.get(block=True, timeout=None)
        p.join(2 * 60)
        if p.exitcode is None:
            p.kill()
        p.join(2 * 60)
        if p.exitcode is None:
            p.terminate()

        if msg == "exception":
            raise ChildProcessError() from value
        assert msg == "eof"
        return value

    @staticmethod
    def _run_func(f, args, kwargs, queue: mp.Queue):
        msg = "eof"
        value = None
        try:
            value = f(*args, **kwargs)
        except BaseException as e:
            msg = "exception"
            value = e
        finally:
            queue.put((msg, value))


def extract_img_metadata(img):
    pixels = img.pixels

    data = {
        "order": pixels.dimension_order.value,
        "big_endian": pixels.big_endian,
        "interleaved": pixels.interleaved,
        "size_x": pixels.size_x,
        "size_y": pixels.size_y,
        "size_z": pixels.size_z,
        "size_t": pixels.size_t,
        "size_c": pixels.size_c,
        "size_x_physical": pixels.physical_size_x,
        "size_y_physical": pixels.physical_size_y,
        "size_z_physical": pixels.physical_size_z,
        "unit_x": pixels.physical_size_x_unit.value,
        "unit_y": pixels.physical_size_y_unit.value,
        "unit_z": pixels.physical_size_z_unit.value,
        "img_type": pixels.type.value,
        "bits": pixels.significant_bits,
        "id": pixels.id,
    }

    return data


def extract_img_annotations(annotations, axis_map: dict) -> dict:
    offset = None
    length = None
    for item in annotations.value.any_elements:
        if item.qname.endswith("Offset"):
            offset = item.attributes
        elif item.qname.endswith("Length"):
            length = item.attributes

    assert len(offset) == len(length)

    data = {}
    for i in range(len(offset)):
        data[f"{axis_map[i]}_offset"] = float(offset[f"Offset_{i}"])

    for i in range(len(length)):
        data[f"{axis_map[i]}_length"] = float(length[f"Length_{i}"])

    return data


def extract_system_annotations(annotations, prefix="xyz-Table ") -> dict:
    data = {}
    for item in annotations.value.any_elements[0].children:
        attrs = item.attributes
        name = attrs["fname"]
        if name.startswith(prefix):
            name = name[len(prefix):]
            if (name.startswith("DevNr") or name.startswith("Flip") or
                    name.startswith("VisualList") or name.startswith("X") or
                    name.startswith("Y") or name.startswith("Z")):
                data[name] = attrs["Value"]

    return data


def extract_ome_table_data(filename):
    ome = from_tiff(
        filename, validate=False,
        parser_kwargs={'config': ParserConfig(fail_on_unknown_properties=False)})

    img = ome.images[0]
    attrs1 = ome.structured_annotations[0]
    attrs2 = ome.structured_annotations[1]
    if attrs1.id != "Annotation:CustomAttributes1":
        raise TypeError(f"Bad ID for attrs 1 {attrs1.id}")
    if attrs2.id != "Annotation:CustomAttributes2":
        raise TypeError(f"Bad ID for attrs 2 {attrs2.id}")

    metadata = extract_img_metadata(img)
    img_annotations = extract_img_annotations(attrs1, axis_map=dim_order)
    sys_annotations = extract_system_annotations(attrs2)

    return metadata, img_annotations, sys_annotations


def extract_num_series_from_names(names: list[str]) -> list[list[int]]:
    # only works for one pattern right now
    if len(names) <= 2:
        raise ValueError("Must have at least 2 files to be able to get number series from their name")

    name_a, name_b, *_ = names

    elements = []
    a_i = 0
    b_i = 0
    start = 0
    in_common = name_a[a_i] == name_b[b_i]
    numbers = "0123456789"
    pat = ""
    if not in_common:
        pat = "([0-9]+)"

    if not in_common and (name_a[a_i] not in numbers or name_b[b_i] not in numbers):
        raise ValueError("Found filename divergence that is not a number")

    while a_i < len(name_a):
        a_inc = 1
        b_inc = 1
        if in_common:
            # we are in a common string element of name
            if name_a[a_i] == name_b[b_i]:
                # still same, just advance
                pass
            else:
                if name_a[a_i] not in numbers or name_b[b_i] not in numbers:
                    raise ValueError("Found filename divergence that is not a number")
                elements.append(name_a[start:a_i])
                in_common = False

        else:
            if name_a[a_i] in numbers and name_b[b_i] in numbers:
                # both are numbers still, just advance
                pass
            elif name_a[a_i] not in numbers and name_b[b_i] not in numbers:
                if name_a[a_i] != name_b[b_i]:
                    raise ValueError("Found filename divergence that is not a number")
                in_common = True
                start = a_i
            elif name_a[a_i] in numbers:
                # only a is still numbering (a is a longer number), b is already at common string part
                # don't advance b, only a
                b_inc = 0
            else:
                # only b is still numbering (b is a longer number), a is already at common string part
                # don't advance a, only b
                a_inc = 0

        a_i += a_inc
        b_i += b_inc

    if in_common:
        elements.append(name_a[start:])

    if len(elements) <= 1:
        raise ValueError("Did not find a numerical pattern in the filenames")

    elements = [e.rstrip("1234567890") for e in elements]
    pat += "([0-9]+)".join([re.escape(e) for e in elements])
    if not in_common:
        pat += "([0-9]+)"

    name_numbers = []
    for name in names:
        m = re.match(pat, name)
        if m is None:
            raise ValueError("Unable to match expected pattern")

        name_numbers.append(list(map(int, m.groups())))

    return name_numbers


def needs_ij(func):
    @wraps(func)
    def callback(obj: "BigStitcherDataset", *args, **kwargs):
        with obj.set_ij():
            return func(obj, *args, **kwargs)

    return callback


def time_me(func=None, name=""):
    def inner(*args, **kwargs):
        nonlocal name
        if not name:
            name = func.__name__
        name = kwargs.pop("description", name)

        ts = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = round(time.perf_counter() - ts)
            secs = elapsed % 60
            elapsed //= 60
            minutes = elapsed % 60
            hours = elapsed // 60
            print(f'{now()}\t"{name}" took {hours:02}:{minutes:02}:{secs:02}')

    if func is None:
        def wrapper(f):
            nonlocal func
            func = f
            return wraps(f)(inner)
        return wrapper

    return wraps(func)(inner)


def replace_str_in_metadata(filename: Path, old: str, new: str):
    data = tifffile.tiffcomment(filename)
    data = data.replace(old, new)
    tifffile.tiffcomment(filename, data.encode("utf8"))


class BigStitcherDataset:

    fused_bounding_box_name: str = "FusedView"

    _initialized_jvm: bool = False

    @staticmethod
    def format_ij_args(args: dict) -> str:
        formatted_args = []
        for key, value in args.items():
            if value is True:
                argument = str(key)
            elif value is False:
                continue
            elif value is None:
                raise NotImplementedError("Conversion for None is not yet implemented")
            else:
                from imagej import jc
                if isinstance(value, jc.ImagePlus):
                    val_str = str(value.getTitle())
                else:
                    val_str = str(value)
                    if not val_str.startswith("[") or not val_str.endswith("]"):
                        val_str = f"[{val_str}]"

                argument = f"{key}={val_str}"

            formatted_args.append(argument)

        return " ".join(formatted_args)

    def __init__(
            self, fiji_path: Path | str, dir_name: str, filename_prefix: str, tiff_filename_pat: str,
            num_x_tiles: int, num_y_tiles: int, channels: list[int],
            tiff_root: str | Path, xml_root: Path | str, fused_root: Path | str, ims_root: Path | str,
            imaris_converter: str | Path, tiff_ext: str = ".ome.tif", input_tiff_in_dir: bool = False,
            y_translation_gear_factor: float = 1,
            xy_voxel_size: float | None = None, z_voxel_size: float | None = None,
            tiles_pos_filename: str | None = None,
    ):
        self.dir_name = dir_name
        self.filename_prefix = filename_prefix
        self.tiff_filename_pat = tiff_filename_pat
        self.tiff_ext = tiff_ext
        self.num_x_tiles = num_x_tiles
        self.num_y_tiles = num_y_tiles
        self.channels = channels
        self.tiff_root = Path(tiff_root)
        self.xml_root = Path(xml_root)
        self.xml_name = f"{filename_prefix}_BS.xml"
        self.fused_root = Path(fused_root)
        self.ims_root = Path(ims_root)
        self.input_tiff_in_dir = input_tiff_in_dir
        self.imaris_converter = imaris_converter
        self.y_translation_gear_factor = y_translation_gear_factor
        self.tiles_pos_filename = tiles_pos_filename
        self.tiles_pos_path = None
        if tiles_pos_filename is not None:
            self.tiles_pos_path = self.input_tiff_dir / tiles_pos_filename
        self.xy_voxel_size = xy_voxel_size
        self.z_voxel_size = z_voxel_size

        self.fiji_path = fiji_path
        self.ij = None

        self.subprocess = ProcAttr(obj_callee=self)

    def populate_voxel_size(self):
        xy_voxel_size = self.xy_voxel_size
        z_voxel_size = self.z_voxel_size

        if xy_voxel_size is None or z_voxel_size is None:
            try:
                images = BigStitcherDataset._get_tiff_images(
                    self.input_tiff_dir, self.filename_prefix + self.tiff_filename_pat + self.tiff_ext, self.num_tiles
                )
                a, _, _ = extract_ome_table_data(str(images[0]))

                if xy_voxel_size is None:
                    assert a["size_x_physical"] == a["size_y_physical"]
                    xy_voxel_size = a["size_x_physical"]
                if z_voxel_size is None:
                    z_voxel_size = a["size_z_physical"]
            except Exception as e:
                raise ValueError(f"Unable to read the tiff image for x,y,z voxel size") from e

        self.xy_voxel_size = xy_voxel_size
        self.z_voxel_size = z_voxel_size

    @property
    def xml_path(self) -> Path:
        return self.xml_root / self.dir_name / self.xml_name

    @property
    def input_tiff_dir(self) -> Path:
        if self.input_tiff_in_dir:
            return self.tiff_root / self.dir_name
        return self.tiff_root

    @property
    def fuse_dir(self):
        return self.fused_root / self.dir_name

    @property
    def ims_dir(self):
        return self.ims_root / self.dir_name

    @property
    def num_tiles(self):
        return self.num_x_tiles * self.num_y_tiles

    @property
    def ims_name(self):
        return f"{self.filename_prefix}_BS.ims"

    def fused_name(self, channel: int):
        return f"{self.filename_prefix}_BS_{channel}.tif"

    @contextmanager
    def set_ij(self):
        if not BigStitcherDataset._initialized_jvm:
            mem = int(psutil.virtual_memory().total / 1024 ** 3 * 0.8 / 3)
            scyjava.config.add_option(f'-Xmx{mem}g')
            BigStitcherDataset._initialized_jvm = True

        ij = None
        try:
            ij = imagej.init(str(self.fiji_path), mode="interactive")
            Prefs = scyjava.jimport('ij.Prefs')
            Prefs.setThreads(int((psutil.cpu_count() - 2) / 2))
            self.ij = ij
            yield ij
        finally:
            self.ij = None
            if ij is not None:
                ij.dispose()

    def get_src_to_resaved_tiff_filenames(
            self, target_dataset: "BigStitcherDataset", start_tile: int = 0,
    ) -> list[tuple[list[Path], str]]:
        n_tiles = self.num_x_tiles * self.num_y_tiles
        n_zeros = math.ceil(math.log10(n_tiles))
        n_total_zeros = math.ceil(math.log10(target_dataset.num_tiles))
        filename_pat = f"{self.filename_prefix}{self.tiff_filename_pat}{self.tiff_ext}"
        n_channels = len(self.channels)

        filenames = []
        for t in range(n_tiles):
            src = []
            for c in range(n_channels):
                curr_fname = filename_pat.format(tile=f"{t:0{n_zeros}}", channel=c)
                img = self.input_tiff_dir / curr_fname
                src.append(img)

            out_tiff = (
                f"{target_dataset.input_tiff_dir / target_dataset.filename_prefix}"
                f"_{t + start_tile:0{n_total_zeros}}"
                f"{target_dataset.tiff_ext}"
            )

            filenames.append((src, out_tiff))

        return filenames

    @needs_ij
    def convert_tiff_to_proper_ome_tiff(
            self, target_dataset: "BigStitcherDataset", start_tile: int = 0, get_metadata_from_each_tile: bool = False,
    ):
        print(f"{now()}\tImporting tiff to Fiji and exporting again to fix metadata for {self.xml_path}")
        target_dataset.input_tiff_dir.mkdir(parents=True, exist_ok=True)
        ij = self.ij

        n_tiles = self.num_x_tiles * self.num_y_tiles
        n_zeros = math.ceil(math.log10(n_tiles))
        n_total_zeros = math.ceil(math.log10(target_dataset.num_tiles))
        filename_pat = f"{self.filename_prefix}{self.tiff_filename_pat}{self.tiff_ext}"
        filenames = self.get_src_to_resaved_tiff_filenames(target_dataset, start_tile)

        for src, out_tiff in tqdm(filenames, total=len(filenames), desc="exporting"):
            img = src[0]

            args = {"open": f"{img}"}
            ij.py.run_plugin("Bio-Formats (Windowless)", BigStitcherDataset.format_ij_args(args))

            args = {
                "save": out_tiff,
                "compression": "Uncompressed",
            }
            ij.py.run_plugin("OME-TIFF...", BigStitcherDataset.format_ij_args(args))

            ij.py.run_plugin("Close All", "")

        if self.tiles_pos_path:
            x = []
            y = []
            lines = self._load_tiles_pos_from_txt(self.tiles_pos_path)
            for t in range(n_tiles):
                name = filename_pat.format(tile=f"{t:0{n_zeros}}", channel=0)
                a, b = lines[name]
                x.append(a)
                y.append(b)
        else:
            x, y = BigStitcherDataset._load_tiles_pos_from_metadata(
                get_metadata_from_each_tile, self.input_tiff_dir, filename_pat, self.num_tiles
            )

        tiles_pos_path = target_dataset.tiles_pos_path
        if not tiles_pos_path.exists():
            with open(tiles_pos_path, "w") as fh:
                fh.write("3\n")

        with open(tiles_pos_path, "a") as fh:
            for t, (a, b) in enumerate(zip(x, y)):
                fh.write(
                    f"{target_dataset.filename_prefix}"
                    f"_{t + start_tile:0{n_total_zeros}}"
                    f"{target_dataset.tiff_ext}"
                    f";;({a},{b},0)\n"
                )

    @time_me(name="Resaving input tiffs")
    def prepare_proper_ome_tiffs_from_datasets(
            self, datasets: list["BigStitcherDataset"], get_metadata_from_each_tile: bool = False
    ):
        start_tile = 0
        for dataset in datasets:
            dataset.convert_tiff_to_proper_ome_tiff(
                self, start_tile, get_metadata_from_each_tile=get_metadata_from_each_tile
            )
            start_tile += dataset.num_tiles

    @time_me(name="Emptying input tiff files")
    def empty_input_tiffs_from_data(
            self, datasets: list["BigStitcherDataset"],
    ):
        start_tile = 0
        for dataset in datasets:
            filenames = dataset.get_src_to_resaved_tiff_filenames(self, start_tile)
            start_tile += dataset.num_tiles

            for src_files, _ in filenames:
                for src in src_files:
                    print(f"{now()}\tEmptying {src}")
                    src.write_bytes(b"")

    @time_me(name="Emptying resaved tiff files")
    def empty_resaved_tiffs_from_data(
            self, datasets: list["BigStitcherDataset"],
    ):
        start_tile = 0
        for dataset in datasets:
            filenames = dataset.get_src_to_resaved_tiff_filenames(self, start_tile)
            start_tile += dataset.num_tiles

            for _, filename in filenames:
                filename = Path(filename)
                print(f"{now()}\tEmptying {filename}")
                filename.write_bytes(b"")

    @time_me(name="Creating xml dataset")
    @needs_ij
    def create_dataset(
            self, automatic_loader: bool = True, load_virtually: bool = True, manually_set_voxel: bool = False
    ):
        self.xml_path.parent.mkdir(parents=True, exist_ok=True)
        ij = self.ij

        if automatic_loader:
            print(f"{now()}\tCreating HDF5 dataset for {self.xml_path}")

            args = {
                "define_dataset": "Automatic Loader (Bioformats based)",
                "project_filename": self.xml_name,
                "path": str(self.input_tiff_dir / f"{self.filename_prefix}*{self.tiff_ext}"),
                "exclude": "10",
                "bioformats_channels_are?": "Channels",
                "pattern_0": "Tiles",
                "move_tiles_to_grid_(per_angle)?": "[Do not move Tiles to Grid (use Metadata if available)]",
                "how_to_store_input_images": "[Re-save as multiresolution HDF5]",
                "load_raw_data_virtually": load_virtually,
                "metadata_save_path": str(self.xml_root / self.dir_name),
                "image_data_save_path": str(self.xml_root / self.dir_name),
                "check_stack_sizes": True,
                "subsampling_factors": "[{ {1,1,1}, {2,2,2}, {4,4,4}, {8,8,8}, {16,16,16}, {32,32,32} }]",
                "hdf5_chunk_sizes": "[{ {64,64,64}, {64,64,64}, {64,64,64}, {64,64,64}, {64,64,64}, {64,64,64} }]",
                "timepoints_per_partition": "1",
                "setups_per_partition": "0",
            }

            if manually_set_voxel:
                if self.xy_voxel_size is None or self.z_voxel_size is None:
                    raise ValueError("Asked to set voxel manually, but voxel sizes not provided")
                args["modify_voxel_size?"] = True
                args["voxel_size_x"] = self.xy_voxel_size
                args["voxel_size_y"] = self.xy_voxel_size
                args["voxel_size_z"] = self.z_voxel_size
                args["voxel_size_unit"] = "µm"
            else:
                args["modify_voxel_size?"] = False

            ij.py.run_plugin("Define Multi-View Dataset", BigStitcherDataset.format_ij_args(args))

            return

        tmp_dset = self.input_tiff_dir / self.xml_name
        print(f"{now()}\tDefining dataset for {tmp_dset}")

        filename_pat = self.tiff_filename_pat.replace("{channel}", "{c}")
        filename_pat = filename_pat.replace("{tile}", "{x}")
        n_tiles = self.num_x_tiles * self.num_y_tiles
        n_zeros = math.ceil(math.log10(n_tiles))

        args = {
            "define_dataset": "Manual Loader (Bioformats based)",
            "project_filename": self.xml_name,
            "multiple_timepoints": "NO (one time-point)",
            "multiple_channels": "YES (one file per channel)",
            "_____multiple_illumination_directions": "NO (one illumination direction)",
            "multiple_angles": "NO (one angle)",
            "multiple_tiles": "YES (one file per tile)",
            "image_file_directory": str(self.input_tiff_dir),
            "image_file_pattern": f"{self.filename_prefix}{filename_pat}{self.tiff_ext}",
            "channels_": ",".join(map(str, range(len(self.channels)))),
            "tiles_": ",".join([f"{i:0{n_zeros}}" for i in range(n_tiles)]),
            "calibration_type": "Same voxel-size for all views",
        }

        if manually_set_voxel:
            if self.xy_voxel_size is None or self.z_voxel_size is None:
                raise ValueError("Asked to set voxel manually, but voxel sizes not provided")
            args["calibration_definition?"] = "User define voxel-size(s)"
            args["pixel_distance_x"] = self.xy_voxel_size
            args["pixel_distance_y"] = self.xy_voxel_size
            args["pixel_distance_z"] = self.z_voxel_size
            args["pixel_unit"] = "um"
        else:
            args["calibration_definition?"] = "Load voxel-size(s) from file(s)"

        ij.py.run_plugin("Define Multi-View Dataset", BigStitcherDataset.format_ij_args(args))

        print(f"{now()}\tCreating HDF5 dataset for {self.xml_path}")
        args = {
            "browse": str(tmp_dset),
            "select": str(tmp_dset),
            "resave_angle": "All angles",
            "resave_channel": "All channels",
            "resave_illumination": "All illuminations",
            "resave_tile": "All tiles",
            "resave_timepoint": "All Timepoints",
            "subsampling_factors": "{ {1,1,1}, {2,2,1} }",
            "hdf5_chunk_sizes": "{ {64,64,64}, {64,64,64} }",
            "timepoints_per_partition": 1,
            "setups_per_partition": 0,
            "export_path": str(self.xml_path),
        }

        ij.py.run_plugin("Resave as HDF5 (local)", BigStitcherDataset.format_ij_args(args))

        tmp_dset.unlink()

    @time_me(name="Editing ome metadata")
    def fix_ome_channel_metadata(self, skip_missing: bool = False):
        n_tiles = self.num_x_tiles * self.num_y_tiles
        n_zeros = math.ceil(math.log10(n_tiles))
        filename_pat = f"{self.filename_prefix}{self.tiff_filename_pat}{self.tiff_ext}"
        print(f"{now()}\tFixing OME metadata for {self.input_tiff_dir / filename_pat}")

        for t in range(n_tiles):
            for c in range(len(self.channels)):
                curr_fname = filename_pat.format(tile=f"{t:0{n_zeros}}", channel=c)
                img = self.input_tiff_dir / curr_fname
                if skip_missing and not img.exists():
                    continue

                try:
                    data = tifffile.tiffcomment(img)
                    data = data.replace('DimensionOrder="XYZCT"', 'DimensionOrder="XYZTC"')
                    for c_num in range(len(self.channels)):
                        data = data.replace("<TiffData>", f'<TiffData FirstC="{c_num}">', 1)

                    tifffile.tiffcomment(img, data.encode("utf8"))
                except ValueError as e:
                    raise ValueError(f"Cannot find metadata in tile {t}, channel {c}") from e

    @staticmethod
    def _load_tiles_pos_from_txt(tiles_path: Path) -> dict[str, tuple[float, float]]:
        with open(tiles_path, "r") as fh:
            lines = {}
            for line in fh.readlines():
                if ";;" in line:
                    name, _, p = line.split(";")
                    a, b, _ = map(float, p.strip().strip("()").split(","))
                    lines[name] = a, b

        return lines

    @staticmethod
    def _load_tiles_pos_from_metadata(
            get_metadata_from_each_tile: bool, tiff_root: Path | str, filename_pat: str, num_tiles: int
    ) -> tuple[list[float], list[float]]:
        x = []
        y = []
        images = BigStitcherDataset._get_tiff_images(tiff_root, filename_pat, num_tiles)

        for i, name in enumerate(images):
            _, b, c = extract_ome_table_data(str(name))
            if get_metadata_from_each_tile:
                x.append(b["x_offset"])
                y.append(b["y_offset"])
            else:
                xvals = list(map(float, c["VisualListX"].strip(",").split(",")))
                yvals = list(map(float, c["VisualListY"].strip(",").split(",")))
                assert len(xvals) == len(yvals)
                assert len(xvals) == len(images)
                x.append(xvals[i])
                y.append(yvals[i])

        return x, y

    @staticmethod
    def _get_tiff_images(tiff_root: Path | str, filename_pat: str, num_tiles: int):
        n_zeros = math.ceil(math.log10(num_tiles))
        images = [tiff_root / filename_pat.format(tile=f"{i:0{n_zeros}}", channel=0) for i in range(num_tiles)]
        return images

    @staticmethod
    def get_tiles_pos_from_image_metadata(
            tiff_root: Path | str, filename_pat: str, num_tiles: int,
            xy_voxel_size: float, tiles_path: Path | None,
            get_metadata_from_each_tile: bool = False,
    ):
        if tiles_path is None:
            x, y = BigStitcherDataset._load_tiles_pos_from_metadata(
                get_metadata_from_each_tile, tiff_root, filename_pat, num_tiles
            )
            return [v / xy_voxel_size for v in x], [v / xy_voxel_size for v in y]

        x = []
        y = []
        images = BigStitcherDataset._get_tiff_images(tiff_root, filename_pat, num_tiles)
        lines = BigStitcherDataset._load_tiles_pos_from_txt(tiles_path)

        for name in images:
            a, b = lines[name.name]
            x.append(a / xy_voxel_size)
            y.append(b / xy_voxel_size)

        return x, y

    @staticmethod
    def get_tiles_pos_from_image_metadata_str(
            tiff_root: Path | str, filename_pat: str, channels: list[int], num_tiles: int,
            y_translation_gear_factor: float, tiles_path: Path | None,
            xy_voxel_size: float,
            get_metadata_from_each_tile: bool = False,
    ) -> str:
        x, y = BigStitcherDataset.get_tiles_pos_from_image_metadata(
            tiff_root, filename_pat, num_tiles, xy_voxel_size, tiles_path, get_metadata_from_each_tile
        )
        x = [v - min(x) for v in x]
        y = [(v - min(y)) * y_translation_gear_factor for v in y]
        y = [-v for v in y]

        s = "dim=3\n"
        i = 0
        for _ in channels:
            for xval, yval in zip(x, y):
                s += f"{i};;({xval}, {yval}, 0)\n"
                i += 1

        return s

    def export_tiles_pos_from_image_metadata(
            self, filename_or_handle: Path | str | IO | None, get_metadata_from_each_tile: bool = False
    ):
        if self.xy_voxel_size is None:
            raise ValueError(f"xy voxel size not provided")

        file_text = self.get_tiles_pos_from_image_metadata_str(
            self.input_tiff_dir,
            f"{self.filename_prefix}{self.tiff_filename_pat}{self.tiff_ext}",
            self.channels,
            self.num_tiles,
            self.y_translation_gear_factor,
            self.tiles_pos_path,
            self.xy_voxel_size,
            get_metadata_from_each_tile,
        )

        if isinstance(filename_or_handle, (Path, str)):
            with open(filename_or_handle, "w") as fh:
                fh.write(file_text)
        elif filename_or_handle is not None:
            filename_or_handle.write(file_text)

        return file_text

    @time_me(name="Setting tile pos from metadata")
    @needs_ij
    def set_tiles_pos_from_image_metadata(self, get_metadata_from_each_tile: bool = False):
        print(f"{now()}\tLoading tile metadata for {self.xml_path}")
        fp = tempfile.NamedTemporaryFile(mode="w", delete_on_close=False)
        filename = Path(fp.name)

        self.export_tiles_pos_from_image_metadata(fp, get_metadata_from_each_tile)

        fp.flush()
        fp.close()

        ij = self.ij

        args = {
            "select": str(self.xml_path),
            "tileconfiguration": str(filename),
            "use_pixel_units": True,
            "keep_metadata_rotation": False,
        }

        ij.py.run_plugin("Load TileConfiguration from File...", BigStitcherDataset.format_ij_args(args))
        filename.unlink()

    @time_me(name="Aligning tiles")
    @needs_ij
    def auto_align_tiles(
            self, min_filter_r=0.9, max_shift_in_x=50, max_shift_in_y=50, max_shift_in_z=50, max_displacement=50,
            align_all_tiles_using_channel: int | None = None, align_all_channels_for_tiles: bool = False,
    ):
        ij = self.ij

        args = {
            "select": str(self.xml_path),
            "process_angle": "All angles",
            "process_channel": "All channels",
            "process_illumination": "All illuminations",
            "process_tile": "All tiles",
            "process_timepoint": "All Timepoints",
            "method": "Phase Correlation",
            "downsample_in_x": 2,
            "downsample_in_y": 2,
            "downsample_in_z": 2,
            "show_expert_algorithm_parameters": True,
            "number_of_peaks_to_check": 5,
            "minimal_overlap": 0,
            "subpixel_accuracy": True,
        }

        if align_all_tiles_using_channel is not None:
            print(
                f"{now()}\tAuto aligning all the tiles across all channels using channel "
                f"{align_all_tiles_using_channel} for {self.xml_path}"
            )
            args["channels"] = f"use Channel {align_all_tiles_using_channel}"
        elif align_all_channels_for_tiles:
            print(f"{now()}\tAuto aligning all the channels for each tile for {self.xml_path}")
            args["show_expert_grouping_options"] = True
            args["how_to_treat_timepoints"] = "treat individually"
            args["how_to_treat_channels"] = "compare"
            args["how_to_treat_illuminations"] = "treat individually"
            args["how_to_treat_angles"] = "treat individually"
            args["how_to_treat_tiles"] = "treat individually"
        else:
            raise ValueError("Either a channel or aligning channels for tiles must be provided")

        ij.py.run_plugin("Calculate pairwise shifts ...", BigStitcherDataset.format_ij_args(args))

        args = {
            "select": str(self.xml_path),
            "filter_by_link_quality": True,
            "min_r": min_filter_r,
            "max_r": 1,
            "filter_by_shift_in_each_dimension": True,
            "max_shift_in_x": max_shift_in_x,
            "max_shift_in_y": max_shift_in_y,
            "max_shift_in_z": max_shift_in_z,
            "filter_by_total_shift_magnitude": True,
            "max_displacement": max_displacement,
        }
        ij.py.run_plugin("Filter pairwise shifts ...", BigStitcherDataset.format_ij_args(args))

        args = {
            "select": str(self.xml_path),
            "process_angle": "All angles",
            "process_channel": "All channels",
            "process_illumination": "All illuminations",
            "process_tile": "All tiles",
            "process_timepoint": "All Timepoints",
            "relative": 2.500,
            "absolute": 3.500,
            "global_optimization_strategy": "One-Round with iterative dropping of bad links",
        }

        if align_all_tiles_using_channel:
            args["fix_group_0-0,"] = True
        elif align_all_channels_for_tiles:
            args["show_expert_grouping_options"] = True
            args["how_to_treat_timepoints"] = "treat individually"
            args["how_to_treat_channels"] = "compare"
            args["how_to_treat_illuminations"] = "treat individually"
            args["how_to_treat_angles"] = "treat individually"
            args["how_to_treat_tiles"] = "treat individually"

            for i in range(self.num_tiles):
                args[f"fix_group_0-{i}"] = True

        ij.py.run_plugin("Optimize globally and apply shifts ...", BigStitcherDataset.format_ij_args(args))

    @time_me(name="Normalizing tile intensities")
    @needs_ij
    def normalize_intensities(self):
        print(f"{now()}\tAdjusting tile intensities for {self.xml_path}")
        ij = self.ij

        args = {
            "select": str(self.xml_path),
            "process_angle": "All angles",
            "process_channel": "All channels",
            "process_illumination": "All illuminations",
            "process_tile": "All tiles",
            "process_timepoint": "All Timepoints",
            "bounding_box": "All Views",
            "downsampling": 32,
            "max_inliers": 10000,
            "affine_intensity": True,
            "offset_only": 0.5,
            "unmodified": 0.5,
        }
        ij.py.run_plugin("Adjust Intensities", BigStitcherDataset.format_ij_args(args))

    @time_me(name="Generating crop area")
    @needs_ij
    def interactively_create_bounding_box(self):
        print(f"{now()}\tCreating bounding box for {self.xml_path}")
        ij = self.ij

        args = {
            "browse": str(self.xml_path),
            "select": str(self.xml_path),
            "process_angle": "All angles",
            "process_channel": "All channels",
            "process_illumination": "All illuminations",
            "process_tile": "All tiles",
            "process_timepoint": "All Timepoints",
            "bounding_box": "Define using the BigDataViewer interactively",
            "bounding_box_name": self.fused_bounding_box_name,
        }
        ij.py.run_plugin("Define Bounding Box", BigStitcherDataset.format_ij_args(args))

    @time_me(name="Fusing tiles")
    @needs_ij
    def fuse_dataset(
            self, use_bounding_box: bool = False, normalize_intensity: bool = False,
    ):
        print(f"{now()}\tFusing {self.xml_path}")

        self.fuse_dir.mkdir(parents=True, exist_ok=True)
        ij = self.ij

        args = {
            "browse": str(self.xml_path),
            "select": str(self.xml_path),
            "process_angle": "All angles",
            "process_channel": "All channels",
            "process_illumination": "All illuminations",
            "process_tile": "All tiles",
            "process_timepoint": "All Timepoints",
            "bounding_box": self.fused_bounding_box_name if use_bounding_box else "All Views",
            "downsampling": 1,
            "interpolation": "Linear Interpolation",
            "fusion_type": "Avg, Blending",
            "pixel_type": "16-bit unsigned integer",
            "interest_points_for_non_rigid": "-= Disable Non-Rigid =-",
            "preserve_original": True,
            "produce": "Each timepoint & channel",
            "fused_image": "Save as (compressed) TIFF stacks",
            "define_input": "Auto-load from input data (values shown below)",
            "output_file_directory": str(self.fuse_dir),
            "filename_addition": self.filename_prefix,
        }
        if normalize_intensity:
            args["adjust_image_intensities"] = True  # should only be used with 32 bit

        ij.py.run_plugin("Image Fusion", BigStitcherDataset.format_ij_args(args))

        file: Path
        for i, channel in enumerate(self.channels):
            new_name = self.fused_name(channel)
            file, = list(Path(self.fuse_dir).glob(f"{glob.escape(self.filename_prefix)}*fused*_ch_{i}.tif"))

            if (Path(self.fuse_dir) / new_name).exists():
                raise ValueError(f"{new_name} already exists")
            file.rename(self.fuse_dir / new_name)

        if (self.fuse_dir / self.xml_path.name).exists():
            raise ValueError(f"{self.fuse_dir / self.xml_path.name} already exists")
        shutil.copy2(self.xml_path, self.fuse_dir)

    def _run_ims_converter(self, tiffs: list[Path], ims_name: str, reading_blocks: int = 20 * 1024):
        ims_path = self.ims_dir / ims_name
        print(f"{now()}\tGenerating Imaris {ims_path}")

        tiffs = list(sorted(tiffs))
        temp_tiffs = [
            f.parent / re.sub(f"(?<=_)({'|'.join(map(str, wavelength_color))})", r"C\g<0>", f.name)
            for f in tiffs
        ]

        for f in temp_tiffs:
            if f.exists():
                raise ValueError(f"{f} already exists")
        if ims_path.exists():
            raise ValueError(f"Ims file {ims_path} already exists")

        for t1, t2 in zip(tiffs, temp_tiffs):
            t1.rename(t2)

        try:
            args = [
                str(self.imaris_converter), "-i", str(temp_tiffs[0]), "-o", str(ims_path), "-ps", str(reading_blocks),
                "-ch", "ColorDefaultHint", "-dcl", "#00ff00", "#ff8800", "#ff0000", "#0000ff",
            ]
            p = subprocess.run(args, capture_output=True, check=True, text=True)
            err = p.stderr
            out = p.stdout
            if out:
                print(f"**Output**:\n{out}")
            if err:
                print(f"Error:\n{err}")
            p.check_returncode()

        finally:
            for t1, t2 in zip(tiffs, temp_tiffs):
                t2.rename(t1)

    @time_me(name="Generating Imaris file")
    def generate_ims(self):
        self.ims_dir.mkdir(parents=True, exist_ok=True)

        files = []
        for channel in self.channels:
            files.append(self.fuse_dir / self.fused_name(channel))
        self._run_ims_converter(files, self.ims_name)

    @time_me(name="Deleting files")
    def delete_files(self, raw_tiffs: bool = False, resaved_tiffs: bool = False, xml: bool = False):
        pass

    def _get_fused_filenames(self):
        files = [
            f"{self.dir_name}/{self.xml_path.name}",
        ]
        for channel in self.channels:
            files.append(f"{self.dir_name}/{self.fused_name(channel)}")
        return files

    def _copy_files(
            self, files: list[tuple[Path, Path]], do_copy: bool, do_verify: bool,
            chunk: int = 1024 ** 3 * 10, verification: str = "size", skip_existing: bool = False,
            overwrite: bool = False, n_retries: int=3, retry_wait: float = 5 * 60.,
    ):
        if do_copy:
            for src, dst in files:
                if dst.exists():
                    if skip_existing:
                        print(f"{now()}\tSkipping existing {src} -> {dst}")
                        continue
                    if overwrite:
                        print(f"{now()}\tOverwriting existing {src} -> {dst}")
                    else:
                        raise ValueError(f"{dst} already exists")

                dst.parent.mkdir(parents=True, exist_ok=True)

                for i in range(n_retries + 1):
                    try:
                        print(f"{now()}\tCopying {src} -> {dst}")
                        shutil.copy2(src, dst)
                        break
                    except OSError as e:
                        if i == n_retries:
                            raise

                        logging.exception(e)
                        print(f"{datetime.datetime.now()}: Trying again {src}")
                        time.sleep(retry_wait)

        if do_verify:
            print("\n\nVerifying files ******************")
            for src, dst in files:
                if not dst.exists():
                    raise ValueError(f"{dst} doesn't exist")

                label = get_drive_name(dst)
                stat = dst.stat()
                src_size = src.stat().st_size

                if verification == "hash":
                    hashes = []
                    for f in (src, dst):
                        m = None

                        for i in range(n_retries + 1):
                            try:
                                m = hashlib.md5()
                                with open(f, "rb") as fh:
                                    line = fh.read(chunk)
                                    while line:
                                        m.update(line)
                                        line = fh.read(chunk)
                                break
                            except OSError as e:
                                if i == n_retries:
                                    raise

                                logging.exception(e)
                                print(f"{datetime.datetime.now()}: Trying again {f}")
                                time.sleep(retry_wait)

                        hashes.append(m.digest())

                    if hashes[0] != hashes[1]:
                        raise ValueError(f"File hash doesn't match for {src}")
                elif verification == "size":
                    if src_size != stat.st_size:
                        raise ValueError(
                            f"File size doesn't match for "
                            f"{convert_bytes(src_size)} {src} -> {convert_bytes(stat.st_size)} {dst}"
                        )
                elif verification == "name":
                    if src.name != dst.name:
                        raise ValueError(
                            f"File names doesn't match for "
                            f"{convert_bytes(src_size)} {src} -> {convert_bytes(stat.st_size)} {dst}"
                        )
                else:
                    raise ValueError(
                        f'Got unknown verification value of "{verification}", '
                        f'valid values are "hash", "size", or "name"'
                    )

                size = convert_bytes(stat.st_size)
                t_str = datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %p %I:%M:S')
                print(f"{now()}\t{size}    {t_str}    {label}    {dst}")

    @time_me(name="Copying raw data")
    def copy_raw(
            self, dest_root: Path | str, do_copy: bool = False, do_verify: bool = False, verification: str = "size",
            skip_existing: bool = False, overwrite: bool = False,
    ):
        dest_root = Path(dest_root)
        files = []
        for item in self.input_tiff_dir.glob("**/*"):
            files.append(
                (
                    item,
                    dest_root / item.relative_to(self.tiff_root)
                )
            )

        self._copy_files(
            files, do_copy, do_verify, verification=verification, skip_existing=skip_existing, overwrite=overwrite
        )

    @time_me(name="Copying fused data")
    def copy_fused(
            self, dest_root: Path | str, do_copy: bool = False, do_verify: bool = False, verification: str = "size",
            skip_existing: bool = False, overwrite: bool = False,
    ):
        dest_root = Path(dest_root)
        files = []
        for name in self._get_fused_filenames():
            files.append(
                (
                    self.fused_root / name,
                    dest_root / name
                )
            )

        self._copy_files(
            files, do_copy, do_verify, verification=verification, skip_existing=skip_existing, overwrite=overwrite
        )

    @time_me(name="Copying ims data")
    def copy_ims(
            self, dest_root: Path | str, do_copy: bool = False, do_verify: bool = False, verification: str = "size",
            skip_existing: bool = False, overwrite: bool = False,
    ):
        dest_root = Path(dest_root)
        files = [
            (self.ims_dir / self.ims_name, dest_root / self.dir_name / self.ims_name),
        ]

        self._copy_files(
            files, do_copy, do_verify, verification=verification, skip_existing=skip_existing, overwrite=overwrite,
        )

    @time_me(name="Processing datasets")
    @staticmethod
    def process_datasets(
            dataset_pairs: list[tuple[list["BigStitcherDataset"] | None, "BigStitcherDataset"]],
            fix_src_ome: bool = False,
            resave_src: bool = False,
            create_xml: bool = False,
            set_tile_pos: bool = False,
            align_channels: bool = False,
            align_tiles_coarse: bool = False,
            align_tiles_coarse_channel: int = 0,
            align_tiles_fine: bool = False,
            align_tiles_fine_channel: int = 1,
            normalize_intensity: bool = False,
            crop_output: bool = False,
            fuse: bool = False,
            generate_ims: bool = False,
            copy_raw_paths: list[Path | str] | None = None,
            copy_fused_paths: list[Path | str] | None = None,
            copy_ims_path: list[Path | str] | None = None,
            copy_raw: bool = False,
            copy_fused: bool = False,
            copy_ims: bool = False,
            verify_raw: bool = False,
            verify_fused: bool = False,
            verify_ims: bool = False,
            verification: str = "size",
            copy_skip_existing: bool = False,
            copy_overwrite: bool = False,
            empty_input_tiffs: bool = False,
            empty_resaved_tiffs: bool = False,
    ):
        if copy_raw:
            for src_dsets, _ in dataset_pairs:
                for dset in src_dsets:
                    for item in copy_raw_paths or []:
                        dset.copy_raw(item, do_copy=True, skip_existing=copy_skip_existing, overwrite=copy_overwrite)

        if verify_raw:
            for src_dsets, _ in dataset_pairs:
                for dset in src_dsets:
                    for item in copy_raw_paths or []:
                        dset.copy_raw(
                            item, do_verify=True, verification=verification, skip_existing=copy_skip_existing,
                            overwrite=copy_overwrite, description="Verifying raw data"
                        )

        if fix_src_ome or resave_src:
            for dsets, _ in dataset_pairs:
                for dset in dsets:
                    dset.populate_voxel_size()

            for src_dsets, dataset in dataset_pairs:
                if fix_src_ome:
                    for dset in src_dsets:
                        dset.fix_ome_channel_metadata()
                if resave_src:
                    dataset.subprocess.prepare_proper_ome_tiffs_from_datasets(src_dsets)

        if create_xml or set_tile_pos:
            for _, dataset in dataset_pairs:
                dataset.populate_voxel_size()

        if (create_xml or set_tile_pos or align_channels or align_tiles_coarse or align_tiles_fine or
                normalize_intensity or crop_output or fuse or empty_input_tiffs or empty_resaved_tiffs):
            for src_dsets, dataset in dataset_pairs:
                if create_xml:
                    dataset.subprocess.create_dataset()
                if set_tile_pos:
                    dataset.subprocess.set_tiles_pos_from_image_metadata()
                if empty_input_tiffs:
                    dataset.empty_input_tiffs_from_data(src_dsets)
                if empty_resaved_tiffs:
                    dataset.empty_resaved_tiffs_from_data(src_dsets)
                if align_channels:
                    dataset.subprocess.auto_align_tiles(
                        max_shift_in_x=30, max_shift_in_y=30, max_shift_in_z=30, max_displacement=30,
                        min_filter_r=0.75,
                        align_all_channels_for_tiles=True,
                        description="Aligning channels for each tile",
                    )
                if align_tiles_coarse:
                    if align_tiles_coarse_channel >= len(dataset.channels):
                        raise ValueError(f"Cannot align with channel {align_tiles_coarse_channel}, not enough channels")
                    dataset.subprocess.auto_align_tiles(
                        max_shift_in_x=300, max_shift_in_y=1000, max_shift_in_z=300, max_displacement=1200,
                        min_filter_r=0.75,
                        align_all_tiles_using_channel=align_tiles_coarse_channel,
                        description=f"Aligning all tiles using channel {align_tiles_coarse_channel}",
                    )
                    dataset.subprocess.auto_align_tiles(
                        max_shift_in_x=300, max_shift_in_y=1000, max_shift_in_z=300, max_displacement=1200,
                        min_filter_r=0.75,
                        align_all_tiles_using_channel=align_tiles_coarse_channel,
                        description=f"Aligning all tiles using channel {align_tiles_coarse_channel}, again",
                    )
                if align_tiles_fine:
                    if align_tiles_fine_channel >= len(dataset.channels):
                        raise ValueError(f"Cannot align with channel {align_tiles_fine_channel}, not enough channels")
                    dataset.subprocess.auto_align_tiles(
                        max_shift_in_x=20, max_shift_in_y=20, max_shift_in_z=20, max_displacement=30,
                        min_filter_r=0.75,
                        align_all_tiles_using_channel=align_tiles_fine_channel,
                        description=f"Aligning all tiles using channel {align_tiles_fine_channel}"
                    )
                if normalize_intensity:
                    dataset.subprocess.normalize_intensities()
                if crop_output:
                    dataset.subprocess.interactively_create_bounding_box()
                if fuse:
                    dataset.subprocess.fuse_dataset(normalize_intensity=True)

        if generate_ims:
            for _, dataset in dataset_pairs:
                dataset.generate_ims()

        if copy_fused:
            for _, dataset in dataset_pairs:
                for item in copy_fused_paths or []:
                    dataset.copy_fused(item, do_copy=True, skip_existing=copy_skip_existing, overwrite=copy_overwrite)

        if copy_ims:
            for _, dataset in dataset_pairs:
                for item in copy_ims_path or []:
                    dataset.copy_ims(item, do_copy=True, skip_existing=copy_skip_existing, overwrite=copy_overwrite)

        if verify_fused:
            for _, dataset in dataset_pairs:
                for item in copy_fused_paths or []:
                    dataset.copy_fused(
                        item, do_verify=True, verification=verification, description="Verifying fused data")

        if verify_ims:
            for _, dataset in dataset_pairs:
                for item in copy_ims_path or []:
                    dataset.copy_ims(item, do_verify=True, verification=verification, description="Verifying ims data")

    @staticmethod
    def make_lavision_datasets(
            names: dict[str, Sequence[int]],
            tiff_drive: Path,
            resaved_tiff_drive: Path,
            xml_drive: Path,
            fused_drive: Path,
            ims_drive: Path,
            per_src_x_tiles: int, num_y_tiles: int,
            fiji_path: Path | str,
            imaris_converter: str | Path,
            tiff_filename_pat="[{tile}]_C0{channel}",
            tiff_ext=".ome.tif",
            y_translation_gear_factor: float = 1,
            src_tiff_root_sub_dir: str | None = "staging",
            resaved_tiff_root_sub_dir: str | None = "staging",
            xml_root_sub_dir: str | None = "xml",
            fused_root_sub_dir: str | None = "fused",
            ims_root_sub_dir: str | None = "ims",
    ):
        src_tiff_root = tiff_drive / src_tiff_root_sub_dir if src_tiff_root_sub_dir else tiff_drive
        resaved_tiff_root = (
                resaved_tiff_drive / resaved_tiff_root_sub_dir) if resaved_tiff_root_sub_dir else resaved_tiff_drive
        xml_root = xml_drive / xml_root_sub_dir if xml_root_sub_dir else xml_drive
        fused_root = fused_drive / fused_root_sub_dir if fused_root_sub_dir else fused_drive
        ims_root = ims_drive / ims_root_sub_dir if ims_root_sub_dir else ims_drive

        n_zeros = math.ceil(math.log10(per_src_x_tiles * num_y_tiles))
        sample_channel = tiff_filename_pat.format(tile=f"{0:0{n_zeros}}", channel=0)
        sample_channel_suffix = f"{sample_channel}{tiff_ext}"

        datasets = []
        for name, chans in names.items():
            src_datasets = []
            sources = {
                p.parent: p.name for p in src_tiff_root.glob(
                    f"**/*{glob.escape(name)}*{glob.escape(sample_channel_suffix)}"
                )
            }

            for src, sample_file_name in sources.items():
                dataset = BigStitcherDataset(
                    dir_name="/".join(src.relative_to(src_tiff_root).parts),
                    filename_prefix=sample_file_name[:-len(sample_channel_suffix)],
                    tiff_filename_pat=tiff_filename_pat, num_x_tiles=per_src_x_tiles, channels=chans,
                    num_y_tiles=num_y_tiles, tiff_root=src_tiff_root, xml_root=xml_root, fused_root=fused_root,
                    ims_root=ims_root, fiji_path=fiji_path, imaris_converter=imaris_converter, tiff_ext=tiff_ext,
                    input_tiff_in_dir=True, y_translation_gear_factor=y_translation_gear_factor,
                    # tiles_pos_filename="tiles.txt",
                )
                src_datasets.append(dataset)
            output_parts = set()
            for src in sources:
                parts = []
                for part in src.relative_to(src_tiff_root).parts:
                    if name in part:
                        part = name
                    parts.append(part)
                output_parts.add("/".join(parts))
            if len(output_parts) != 1:
                raise ValueError(
                    f"Expected to find exactly one root directory for {name} in {src_tiff_root}. Found {output_parts}"
                )

            dataset = BigStitcherDataset(
                dir_name=list(output_parts)[0],
                filename_prefix=name,
                tiff_filename_pat="_{tile}", num_x_tiles=per_src_x_tiles * len(src_datasets), channels=chans,
                num_y_tiles=num_y_tiles, tiff_root=resaved_tiff_root, xml_root=xml_root, fused_root=fused_root,
                ims_root=ims_root, fiji_path=fiji_path, imaris_converter=imaris_converter, tiff_ext=tiff_ext,
                input_tiff_in_dir=True, y_translation_gear_factor=y_translation_gear_factor,
                tiles_pos_filename="tiles.txt",
            )

            datasets.append((src_datasets, dataset))

        return datasets


if __name__ == '__main__':
    mp.set_start_method('spawn')
    # measure_memory()

    c2 = [488, 640]
    c3 = [488, 561, 640]
    lavision_datasets = BigStitcherDataset.make_lavision_datasets(
        # # aws
        # names={
        #     "MF1_267F_W": c2,
        # },
        # tiff_drive=Path(r"H:\imaging"),
        # resaved_tiff_drive=Path(r"I:\imaging"),
        # xml_drive=Path(r"K:\imaging"),
        # fused_drive=Path(r"H:\imaging"),
        # ims_drive=Path(r"I:\imaging"),

        # # aws
        # names={
        # },
        # tiff_drive=Path(r"L:\imaging"),
        # resaved_tiff_drive=Path(r"M:\imaging"),
        # xml_drive=Path(r"N:\imaging"),
        # fused_drive=Path(r"L:\imaging"),
        # ims_drive=Path(r"M:\imaging"),

        per_src_x_tiles=3,
        num_y_tiles=4,
        fiji_path=Path(r'C:\Users\CPLab\Fiji.app'),
        imaris_converter=Path(r"C:\Program Files\Bitplane\ImarisFileConverter 10.2.0\ImarisConvert.exe"),
        y_translation_gear_factor=1 + 24 / 50,
    )

    # BigStitcherDataset.process_datasets(
    #     lavision_datasets,
    #     copy_raw_paths=[r"Z:\2025_ALL_USER_IMAGES_HERE\ClelandThomas\data"],
    #     verify_raw=True,
    #     verification="hash",
    # )
    # BigStitcherDataset.process_datasets(
    #     lavision_datasets,
    #     copy_raw_paths=[r"O:\Yidan\Brains"],
    #     copy_raw=True,
    #     verify_raw=True,
    #     verification="hash",
    #     copy_skip_existing=False,
    #     # copy_overwrite=True,
    # )
    # BigStitcherDataset.process_datasets(
    #     lavision_datasets,
    #     fix_src_ome=True,
    #     resave_src=True,
    #     create_xml=True,
    #     set_tile_pos=True,
    #     align_channels=True,
    #     align_tiles_coarse=True,
    # )
    # BigStitcherDataset.process_datasets(
    #     lavision_datasets,
    #     align_tiles_coarse=True,
    # )

    # BigStitcherDataset.process_datasets(
    #     lavision_datasets,
    #     align_tiles_fine=True,
    #     normalize_intensity=True,
    # )
    # BigStitcherDataset.process_datasets(
    #     lavision_datasets,
    #     crop_output=True,
    # )
    # BigStitcherDataset.process_datasets(
    #     lavision_datasets,
    #     empty_input_tiffs=True,
    #     empty_resaved_tiffs=True,
    # )
    # BigStitcherDataset.process_datasets(
    #     lavision_datasets,
    #     fuse=True,
    #     generate_ims=True,
    # )
    BigStitcherDataset.process_datasets(
        lavision_datasets,
        copy_fused_paths=[r"U:\BrainClearing\Brain_Images\Processed\fused", r"Q:\Yidan\Brains"],
        copy_ims_path=[r"U:\BrainClearing\Brain_Images\Processed\ims"],
        copy_fused=True,
        copy_ims=True,
        verify_fused=True,
        verify_ims=True,
        # copy_overwrite=True,
        # copy_skip_existing=True,
    )
    BigStitcherDataset.process_datasets(
        lavision_datasets,
        copy_fused_paths=[r"Q:\Yidan\Brains"],
        verify_fused=True,
        verification="hash"
    )
