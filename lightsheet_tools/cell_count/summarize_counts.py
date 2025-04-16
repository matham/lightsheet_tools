import csv
import re
from pprint import pprint
from copy import deepcopy
from typing import Any
from pathlib import Path
from collections import deque
from collections.abc import Sequence


class RegionNode:

    name: str = ""

    acronym: str = ""

    region_id: int = -1

    parent: "RegionNode" = None

    children: list["RegionNode"] = None

    num_sub_children: int = 0

    rgb: tuple[int, int, int] = (0, 0, 0)

    def __init__(self, name: str, acronym: str, region_id: int, rgb: tuple[int, int, int] = (0, 0, 0), **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.acronym = acronym
        self.region_id = region_id
        self.children = []
        self.rgb = rgb

    @property
    def is_leaf(self):
        return not len(self.children)

    def locate_name(self, name: str) -> "RegionNode":
        return self._locate("name", name)

    def locate_acronym(self, acronym: str) -> "RegionNode":
        return self._locate("acronym", acronym)

    def locate_id(self, region_id: int) -> "RegionNode":
        return self._locate("region_id", region_id)

    def _locate(self, attr: str, value) -> "RegionNode":
        for node in self.iter_children_breath_first(include_self=True):
            if getattr(node, attr) == value:
                return node

        raise ValueError(f"Could not find region {value}")

    def iter_children_breath_first(self, include_self: bool):
        if include_self:
            queue = deque([self])
        else:
            queue = deque(self.children)

        while queue:
            item = queue.popleft()
            yield item
            queue.extend(item.children)

    def iter_children_depth_first_parent_first(self, include_self: bool):
        if include_self:
            queue = deque([self])
        else:
            queue = deque(self.children)

        while queue:
            item = queue.popleft()
            yield item
            queue.extendleft(item.children)

    def iter_children_depth_first_parent_last(self, include_self: bool):
        added_children = set()
        if include_self:
            queue = deque([self])
        else:
            queue = deque(self.children)

        while queue:
            item = queue.popleft()
            if item not in added_children:
                queue.appendleft(item)
                queue.extendleft(item.children)
                added_children.add(item)
                continue

            yield item

    def add_child(self, child: "RegionNode") -> None:
        self.children.append(child)
        child.parent = self

    @staticmethod
    def populate_num_children(root: "RegionNode") -> None:
        for node in root.iter_children_depth_first_parent_last(include_self=True):
            node.num_sub_children = sum(n.num_sub_children for n in node.children) + len(node.children)

    @classmethod
    def load_region_graph(cls, filename: Path) -> "RegionNode":
        with open(filename, "r") as fh:
            reader = csv.reader(fh, delimiter=",", quotechar='"')
            rows = [r for r in reader]

        nodes_ids = {}
        nodes_parent = {}
        root = None
        for row in rows[1:]:
            region_id, name, acronym, r, g, b, _, parent_id = row
            if not region_id:
                continue

            region_id, r, g, b, parent_id = map(int, (region_id, r, g, b, parent_id))

            node = cls(name=name, acronym=acronym, region_id=region_id, rgb=(r, g, b))
            nodes_ids[region_id] = node

            if root is None:
                root = node
            else:
                nodes_parent[node] = parent_id

        for node, parent_id in nodes_parent.items():
            nodes_ids[parent_id].add_child(node)

        return root

    def __str__(self):
        return repr(self)

    def get_regions(
            self, names: Sequence[str] = (), acronyms: Sequence[str] = (), ids: Sequence[int] = (),
            order: str = "dfspf", include_roots: bool = True,
    ) -> list["RegionNode"]:
        roots = [self.locate_name(name) for name in names]
        roots += [self.locate_acronym(a) for a in acronyms]
        roots += [self.locate_id(i) for i in ids]

        nodes = []
        for root in roots:
            match order:
                case "bfs":
                    it = root.iter_children_breath_first(include_self=include_roots)
                case "dfspf":
                    it = root.iter_children_depth_first_parent_first(include_self=include_roots)
                case "dfspl":
                    it = root.iter_children_depth_first_parent_last(include_self=include_roots)
                case _:
                    raise ValueError(f"Unrecognizable order {order}, valid values are bfs, dfspf, dfspl")

            nodes.extend(it)

        return nodes


class CountedRegionNode(RegionNode):

    voxels: int = 0

    volume_mm: float = 0

    cell_count: int = 0

    cell_density: float = 0

    def __repr__(self):
        tp = "Leaf" if self.is_leaf else "Node"
        return (
            f"<{tp} {self.acronym} "
            f"({self.region_id}). "
            f"{self.cell_count} cells / "
            f"{self.volume_mm} mm3 "
            f"{id(self)}@"
            f"{self.__class__.__qualname__}>"
        )


class SubjectRegions:

    subject_id: str = ""

    channel: int = 0

    count_atlas: str = ""

    count_algo: str = ""

    root_region: CountedRegionNode = None

    tags: dict[str, Any] = {}

    def __init__(
            self, subject_id: str, channel: int, count_atlas: str, count_algo: str, tags: dict[str, Any] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.subject_id = subject_id
        self.channel = channel
        self.count_atlas = count_atlas
        self.count_algo = count_algo
        self.tags = tags or {}

    @classmethod
    def parse_count_csv(cls, filename: Path, root: CountedRegionNode, exclude_regions: set[int]) -> CountedRegionNode:
        root = deepcopy(root)

        with open(filename, "r") as fh:
            it = iter(csv.reader(fh, delimiter=",", quotechar='"'))
            next(it)

            for id_, _, _, voxels, vol, count, density in it:
                if not id_:
                    continue

                id_ = int(float(id_))
                if id_ in exclude_regions:
                    continue

                node = root.locate_id(id_)
                node.voxels = int(float(voxels))
                node.volume_mm = float(vol)
                node.cell_count = int(float(count))
                node.cell_density = float(density)

        return root

    @classmethod
    def parse_directory(
            cls, root_dir: Path, root: CountedRegionNode, exclude_regions: set[int],
    ) -> list["SubjectRegions"]:
        subjects = []
        for filename in root_dir.glob("*.csv"):
            m = re.match("cell_counts_([a-zA-Z0-9_.]+?)_([0-9]+)_([a-zA-Z0-9_.]+?)_density.csv", filename.name)
            if m is None:
                raise ValueError(f"Could not match {filename}")

            try:
                name, channel, atlas = m.groups()
                root = cls.parse_count_csv(filename, root, exclude_regions)
            except Exception as e:
                raise ValueError(f"{filename} failed to parse") from e
            subject = cls(subject_id=name, channel=int(channel), count_atlas=atlas, count_algo="count_gd")
            subject.root_region = root
            subjects.append(subject)

        return subjects


if __name__ == "__main__":
    regions_csv = Path(__file__).parent / "UnifiedAtlas_Label_ontology_v2.csv"
    count_root = Path(__file__).parent / "counts"

    root_region = CountedRegionNode.load_region_graph(regions_csv)
    subjects = SubjectRegions.parse_directory(count_root, root_region, exclude_regions={2285, 2278, 728, 997})

    regions = subjects[0].root_region.get_regions(acronyms=("MOB", ), include_roots=True)
    pprint(list(regions))
