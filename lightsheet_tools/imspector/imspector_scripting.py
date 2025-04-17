import pyautogui as pag
import pyscreeze
import pygetwindow
import ctypes
from PIL import Image
import time
import datetime
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager

X_LEFT_UM = -5592.97
X_RIGHT_UM = 4415.16

Y_BOTTOM_UM = -4711.17
Y_TOP_UM = 4995.70

SUBJECT_NAME = "MF1_218M_W"


RESOURCE_ROOT = Path(__file__).parent / "resources"


class ImgResources:

    filenames: dict[str, str] = {
        "autosave-main-v1": "autosave-main-v1.png",
        "autosave-main-v2": "autosave-main-v2.png",
        "autosave-main-v3": "autosave-main-v3.png",
        "autosave-ok": "autosave-ok.png",
        "autosave-prefix": "autosave-prefix.png",
        "autosave-location": "autosave-location.png",
        "position-active-v1": "position-active-v1.png",
        "position-active-v2": "position-active-v2.png",
        "position-add-v1": "position-add-v1.png",
        "position-add-v2": "position-add-v2.png",
        "position-close": "position-close.png",
        "position-current-v1": "position-current-v1.png",
        "position-current-v2": "position-current-v2.png",
        "record-off": "record-off.png",
        "sheet-left-off": "sheet-left-off.png",
        "sheet-left-on": "sheet-left-on.png",
        "sheet-right-off": "sheet-right-off.png",
        "sheet-right-on": "sheet-right-on.png",
        "tiles-clear-all-v1": "tiles-clear-all-v1.png",
        "tiles-clear-all-v2": "tiles-clear-all-v2.png",
        "tiles-move-left": "tiles-move-left.png",
        "tiles-move-right": "tiles-move-right.png",
        "tiles-x": "tiles-x.png",
        "tiles-y": "tiles-y.png",
    }

    versions: dict[str, tuple[str, ...]] = {
        "autosave-main": ("autosave-main-v1", "autosave-main-v2", "autosave-main-v3"),
        "position-active": ("position-active-v1", "position-active-v2"),
        "position-add": ("position-add-v1", "position-add-v2"),
        "position-current": ("position-current-v1", "position-current-v2"),
        "sheet-left": ("sheet-left-off", "sheet-left-on"),
        "sheet-right": ("sheet-right-on", "sheet-right-off"),
        "tiles-clear-all": ("tiles-clear-all-v1", "tiles-clear-all-v2"),
    }

    def get(self, name: str) -> tuple[str, ...]:
        if name in self.versions:
            names = self.versions[name]
        else:
            names = [name]

        return tuple(str(RESOURCE_ROOT / self.filenames[n]) for n in names)


@pyscreeze.requiresPyGetWindow
def activate_window(window_title):
    matching_windows = pygetwindow.getWindowsWithTitle(window_title)
    if len(matching_windows) == 0:
        raise ValueError(f'Could not find a window with {window_title} in the title')
    elif len(matching_windows) > 1:
        raise ValueError(f'Found multiple windows with {window_title} in the title')

    win = matching_windows[0]
    win.activate()
    time.sleep(2)

    return win.left, win.top, win.width, win.height


class ImSpectorOps:

    resources = ImgResources()

    pos_cache: None | dict = None

    def __init__(self, window_title: str):
        super().__init__()
        self.window_title = window_title
        self.region = None

    @contextmanager
    def enable_pos_cache(self):
        try:
            self.pos_cache = {}
            yield
        finally:
            self.pos_cache = None

    def locate_on_screen(
            self, item_name: str, wait_timeout: float = 5, x_offset_factor: float = 0, y_offset_factor: float = 0
    ) -> pag.Point:
        cache_key = item_name, x_offset_factor, y_offset_factor
        if self.pos_cache is not None and cache_key in self.pos_cache:
            return self.pos_cache[cache_key]

        ts = time.perf_counter()
        while time.perf_counter() - ts < wait_timeout:
            for n in self.resources.get(item_name):
                try:
                    loc = pag.locateOnScreen(n, confidence=.95)
                except pag.ImageNotFoundException:
                    continue

                center = pag.center(loc)
                img = Image.open(n)
                w, h = img.size
                p = pag.Point(center.x + x_offset_factor * w, center.y + y_offset_factor * h)

                if self.pos_cache is not None:
                    self.pos_cache[cache_key] = p
                return p

            time.sleep(1)

        raise pag.ImageNotFoundException(f"Didn't find {item_name}")

    def move_to_location(
            self, item_name: str, wait_timeout: float = 5, x_offset_factor: float = 0,
            y_offset_factor: float = 0
    ):
        pos = self.locate_on_screen(item_name, wait_timeout, x_offset_factor, y_offset_factor)
        pag.moveTo(*pos)
        return pos

    def press_button(
            self, item_name: str, post_delay: float = 0, wait_timeout: float = 5, x_offset_factor: float = 0,
            y_offset_factor: float = 0
    ):

        pos = self.move_to_location(item_name, wait_timeout, x_offset_factor, y_offset_factor)
        time.sleep(0.1)
        pag.click(*pos)

        if post_delay:
            time.sleep(post_delay)

    def type_text(
            self, item_name: str, value: str, post_delay: float = 0, wait_timeout: float = 5,
            x_offset_factor: float = 0, y_offset_factor: float = 0
    ):
        self.press_button(item_name, .5, wait_timeout, x_offset_factor, y_offset_factor)
        pag.hotkey("ctrl", "a", interval=0.1)
        time.sleep(0.1)
        pag.write(value, interval=0.1)

        if post_delay:
            time.sleep(post_delay)

    def ensure_window_active(self):
        activate_window(self.window_title)
        time.sleep(1)

    def add_tiles(self, tiles: list[tuple[int, int]]):
        self.press_button("tiles-clear-all", post_delay=1)
        self.press_button("position-active", post_delay=1)

        with self.enable_pos_cache():
            for x, y in tiles:
                self.press_button("position-add", post_delay=1)
                self.press_button("tiles-move-left", post_delay=1)
                self.type_text("tiles-x", str(int(x)), post_delay=1, y_offset_factor=1)
                self.type_text("tiles-y", str(int(y)), post_delay=1, y_offset_factor=1)

        self.press_button("position-close", post_delay=1)

    def wait_for_record_end(self, timeout: int = 24 * 60 * 60):
        self.locate_on_screen("record-off", timeout)

    def update_autosave(self, prefix: str, root: str | Path):
        self.press_button("autosave-main", post_delay=1)
        self.type_text("autosave-prefix", prefix, post_delay=1, x_offset_factor=1)
        self.type_text("autosave-location", str(root), post_delay=1, x_offset_factor=1)
        self.press_button("autosave-ok", post_delay=1)

    def set_experiment_side(self, root: str | Path, file_prefix: str, side: str):
        side_short = {"left": "L", "right": "R"}[side]
        self.ensure_window_active()

        self.press_button(f"sheet-{side}", post_delay=1)
        pos = getattr(tiler, f"{side}_tiles_pos")
        self.add_tiles([(int(x), int(y)) for x, y in pos])
        self.update_autosave(f"{file_prefix}_{side_short}", root)

    def run_experiment(
            self, tiler: "SampleTiling", root: str | Path, file_prefix: str, add_data_subdir: bool = True
    ):
        if ctypes.WinDLL("User32.dll").GetKeyState(0x14):
            pag.alert("Caps lock is ON. Turn it OFF and run again.", "Capslock is ON")
            return

        msg = f"Name: {file_prefix}_(L/R)\n"
        msg += f"OVERLAP (px): {int(tiler.overlap_x_pixels)} X {int(tiler.overlap_y_pixels)}\n\n"
        msg += f"{tiler}"
        selection = pag.confirm(text=msg, title="Confirm experiment", buttons=['OK', 'Abort'])
        if selection == "Abort":
            return

        if add_data_subdir:
            now = datetime.datetime.now()
            root = Path(root) / now.strftime("%Y%m%d")
            if not root.exists():
                root.mkdir(parents=True)

        self.set_experiment_side(root, file_prefix, "left")

        self.press_button("record-off", post_delay=1)
        self.move_to_location("autosave-main")
        self.wait_for_record_end()
        time.sleep(10)

        self.set_experiment_side(root, file_prefix, "right")

        self.press_button("record-off", post_delay=1)
        self.move_to_location("autosave-main")


@dataclass
class SampleTiling:

    total_mag: float = 1

    num_x_tiles: int = 1

    num_y_tiles: int = 1

    x_tile_side_split: tuple[int, int] = (1, 1)

    y_gear_ratio: float = 0

    uses_tank: bool = False

    pixel_size_base_um: float = 6.5

    num_x_pixels: int = 2160

    num_y_pixels: int = 2560

    max_x_translation_mm: float = 11.9

    max_y_translation_mm: float = 10.8

    x_left_um: float = 0

    x_right_um: float = 1

    y_bottom_um: float = 0

    y_top_um: float = 0

    @property
    def pixel_size_um(self) -> float:
        return self.pixel_size_base_um / self.total_mag

    @property
    def tile_width_um(self) -> float:
        return self.pixel_size_um * self.num_x_pixels

    @property
    def tile_height_um(self) -> float:
        return self.pixel_size_um * self.num_y_pixels

    @property
    def full_x_imaged_size_um(self) -> float:
        return self.x_right_um - self.x_left_um + self.tile_width_um

    @property
    def full_y_imaged_size_um(self) -> float:
        multiplier = 1
        if self.uses_tank:
            multiplier += self.y_gear_ratio

        return (self.y_top_um - self.y_bottom_um) * multiplier + self.tile_height_um

    @property
    def overlap_x_pixels(self) -> float:
        remaining_width_um = self.num_x_tiles * self.tile_width_um - self.full_x_imaged_size_um
        remaining_width_px = remaining_width_um / self.pixel_size_um
        overlap_px = remaining_width_px / (self.num_x_tiles - 1)
        return overlap_px

    @property
    def overlap_y_pixels(self) -> float:
        remaining_height_um = self.num_y_tiles * self.tile_height_um - self.full_y_imaged_size_um
        remaining_height_px = remaining_height_um / self.pixel_size_um
        overlap_px = remaining_height_px / (self.num_y_tiles - 1)
        return overlap_px

    @property
    def overlap_x_um(self) -> float:
        return self.overlap_x_pixels * self.pixel_size_um

    @property
    def overlap_y_um(self) -> float:
        return self.overlap_y_pixels * self.pixel_size_um

    @property
    def x_tiles_pos_um(self) -> list[float]:
        tiles = []
        for i in range(self.num_x_tiles):
            x = self.x_left_um + i * (self.num_x_pixels - self.overlap_x_pixels) * self.pixel_size_um
            tiles.append(x)
        return tiles

    @property
    def y_tiles_pos_um(self) -> list[float]:
        multiplier = 1
        if self.uses_tank:
            multiplier += self.y_gear_ratio

        tiles = []
        for i in range(self.num_y_tiles):
            y = self.y_bottom_um + i * (self.num_y_pixels - self.overlap_y_pixels) * self.pixel_size_um / multiplier
            tiles.append(y)
        return tiles

    @property
    def left_tiles_pos(self) -> list[tuple[float, float]]:
        if sum(self.x_tile_side_split) != self.num_x_tiles:
            raise ValueError(f"split {self.x_tile_side_split} doesn't match num x tiles {self.num_x_tiles}")

        pos = []
        for y in self.y_tiles_pos_um:
            for x in self.x_tiles_pos_um[:self.x_tile_side_split[0]]:
                pos.append((x, y))

        return pos

    @property
    def right_tiles_pos(self) -> list[tuple[float, float]]:
        if sum(self.x_tile_side_split) != self.num_x_tiles:
            raise ValueError(f"split {self.x_tile_side_split} doesn't match num x tiles {self.num_x_tiles}")

        pos = []
        for y in self.y_tiles_pos_um:
            for x in self.x_tiles_pos_um[self.x_tile_side_split[0]:]:
                pos.append((x, y))

        return pos

    @property
    def tiles_pos(self) -> list[tuple[float, float]]:
        pos = []
        for y in self.y_tiles_pos_um:
            for x in self.x_tiles_pos_um:
                pos.append((x, y))
        return pos

    def __str__(self):
        s = f"Magnification: {self.total_mag}X\n"
        s += f"Tiles (WxH): {self.num_x_tiles} X {self.num_y_tiles}\n"
        s += f"ROI (pixels): {self.num_x_pixels} X {self.num_y_pixels}\n"
        s += f"Full brain (mm): {int(self.full_x_imaged_size_um) / 1000} X {int(self.full_y_imaged_size_um) / 1000}\n"
        s += f"Tile overlap (pixels): {int(self.overlap_x_pixels)} X {int(self.overlap_y_pixels)}\n"
        s += f"# left/right tiles: {self.x_tile_side_split[0]}, {self.x_tile_side_split[1]}\n"
        s += f"Using tank: {self.uses_tank}\n"
        return s


if __name__ == "__main__":
    ops = ImSpectorOps(window_title="ImSpector - ")
    tiler = SampleTiling(
        total_mag=3.2,
        y_gear_ratio=24 / 50,
        uses_tank=True,
        num_x_pixels=1621 - 540,
        num_x_tiles=6,
        num_y_tiles=4,
        x_tile_side_split=(3, 3),
        x_left_um=X_LEFT_UM,
        x_right_um=X_RIGHT_UM,
        y_bottom_um=Y_BOTTOM_UM,
        y_top_um=Y_TOP_UM,
    )
    try:
        ops.run_experiment(
            tiler,
            r"D:\ALL_USER_DATA_HERE\Cleland\autosave",
            SUBJECT_NAME,
        )
    except:
        pag.alert("Recording failed")
        raise
    else:
        pag.alert("Recording finished")