import csv
import numpy as np
import re
from pprint import pprint
from functools import partial
from copy import deepcopy
from typing import Any
from pathlib import Path
from collections import deque, defaultdict
from itertools import product
from collections.abc import Sequence
import matplotlib.pyplot as plt


class RegionNode:

    name: str = ""

    acronym: str = ""

    region_id: int = -1

    parent: "RegionNode" = None

    children: list["RegionNode"] = None

    num_sub_children: int = 0

    rgb: tuple[int, int, int] = (0, 0, 0)

    _region_id_map: dict[int, "RegionNode"] = None
    # invariant: node and all its children always share the same _region_id_map instance

    def __init__(self, name: str, acronym: str, region_id: int, rgb: tuple[int, int, int] = (0, 0, 0), **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.acronym = acronym
        self.region_id = region_id
        self.children = []
        self.rgb = rgb
        self._region_id_map = {region_id: self}

    @property
    def is_leaf(self):
        return not len(self.children)

    def locate_name(self, name: str) -> "RegionNode":
        return self._locate("name", name)

    def locate_acronym(self, acronym: str) -> "RegionNode":
        return self._locate("acronym", acronym)

    def locate_id(self, region_id: int) -> "RegionNode":
        return self._region_id_map[region_id]

    def locate_node(self, node: "RegionNode") -> "RegionNode":
        return self._region_id_map[node.region_id]

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

        if self == child:
            raise ValueError(f"Tried to add self to itself: {self}")

        # if we ever remove children, we'd need to clear this
        self._region_id_map.update(child._region_id_map)
        for node in child.iter_children_breath_first(include_self=True):
            node._region_id_map = self._region_id_map

    @staticmethod
    def populate_num_children(root: "RegionNode") -> None:
        for node in root.iter_children_depth_first_parent_last(include_self=True):
            node.num_sub_children = sum(n.num_sub_children for n in node.children) + len(node.children)

    @classmethod
    def load_region_graph_from_ontology(cls, filename: Path) -> "RegionNode":
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
            if nodes_ids[parent_id] == node:
                # don't add node to itself - root nodes are listed as children of itself
                continue
            nodes_ids[parent_id].add_child(node)

        cls.populate_num_children(root)

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

    def deep_copy_graph(self) -> "RegionNode":
        return deepcopy(self)


class CountedRegionNode(RegionNode):

    voxels: float = 0

    volume_mm: float = 0

    cell_count: float = 0

    cell_density: float = 0

    num_samples: int = 1

    def __repr__(self):
        tp = "Leaf" if self.is_leaf else "Node"
        return (
            f"<{tp} {self.acronym} "
            f"({self.region_id}). "
            f"{self.cell_count:0.0f} cells / "
            f"{self.volume_mm:0.1f} mm3 "
            f"(N={self.num_samples}). "
            f"{id(self)}@"
            f"{self.__class__.__qualname__}>"
        )

    def __eq__(self, other):
        if not isinstance(other, RegionNode):
            return False

        return self.region_id == other.region_id and self.acronym == other.acronym and self.name == other.name

    def __hash__(self):
        return id(self)

    def copy_summary(self, *nodes, summary_func = np.mean) -> None:
        self.num_samples = n = len(nodes)
        self.voxels = summary_func([s.voxels for s in nodes]) if n else 0
        self.volume_mm = summary_func([s.volume_mm for s in nodes]) if n else 0
        self.cell_count = summary_func([s.cell_count for s in nodes]) if n else 0
        self.cell_density = summary_func([s.cell_density for s in nodes]) if n else 0


class SubjectRegions:

    root_region: CountedRegionNode = None

    tags: dict[str, Any] = {}

    def __init__(
            self, tags: dict[str, Any] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.tags = tags or {}

    @classmethod
    def parse_count_csv(cls, filename: Path, root: CountedRegionNode, exclude_regions: set[int]) -> CountedRegionNode:
        root = root.deep_copy_graph()

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
    def parse_subject_metadata(cls, filename: Path) -> dict[str, dict[int, dict[str, Any]]]:
        with open(filename, "r") as fh:
            reader = csv.reader(fh, delimiter=",")
            lines = list(reader)

        metadata = defaultdict(dict)
        for name, channel, strain, condition, cell_label, sex in lines[1:]:
            channel = int(channel)
            metadata[name][channel] = {
                "strain": strain,
                "condition": condition,
                "cell_label": cell_label,
                "sex": sex,
            }

        return metadata

    @classmethod
    def parse_directory(
            cls, root_dir: Path, root: CountedRegionNode, exclude_regions: set[int],
            subjects_metadata: dict[str, dict[int, dict[str, Any]]],
    ) -> list["SubjectRegions"]:
        subjects = []
        for filename in root_dir.glob("*.csv"):
            m = re.match(
                "cell_counts_([a-zA-Z0-9_.]+?)_([0-9]+)_([a-zA-Z0-9_.]+?)_([0-9]+)um(_\\w+?)?_density.csv",
                filename.name
            )
            if m is None:
                raise ValueError(f"Could not match {filename}")

            try:
                name, channel, atlas, resolution, others = m.groups()
                channel = int(channel)
                tags = {
                    "name": name,
                    "channel": int(channel),
                    "atlas": atlas,
                    "resolution": int(resolution),
                    "algorithm": others,
                }
                tags.update(subjects_metadata[name][channel])

                root = cls.parse_count_csv(filename, root, exclude_regions)
            except Exception as e:
                raise ValueError(f"{filename} failed to parse") from e

            subject = cls(tags=tags)
            subject.root_region = root
            subjects.append(subject)

        return subjects

    @classmethod
    def select_subjects(cls, subjects: list["SubjectRegions"], **tags) -> list["SubjectRegions"]:
        selected = []
        for subject in subjects:
            for key, value in tags.items():
                if isinstance(subject.tags[key], str):
                    if subject.tags[key].lower() != value.lower():
                        break
                else:
                    if subject.tags[key] != value:
                        break
            else:
                selected.append(subject)

        return selected

    @classmethod
    def summarize_subjects(
            cls, subjects: list["SubjectRegions"], summary_func = np.mean, **tags
    ) -> CountedRegionNode:
        selected = cls.select_subjects(subjects, **tags)
        if not selected:
            raise ValueError(f"No subjects found matching all tags: {tags}")

        locators = [s.root_region.locate_node for s in selected]
        root = selected[0].root_region.deep_copy_graph()
        for node in root.iter_children_breath_first(True):
            sources = [locate(node) for locate in locators]

            node.copy_summary(*sources, summary_func=summary_func)

        return root

    @staticmethod
    def build_tags_product(**tags_list: list) -> list[dict]:
        tags_groups = []
        keys = list(tags_list.keys())
        for values in product(*(tags_list[key] for key in keys)):
            assert len(keys) == len(values)
            tags = {key: value for key, value in zip(keys, values)}
            tags_groups.append(tags)

        return tags_groups

    @staticmethod
    def build_tags_zip(**tags_list: list) -> list[dict]:
        tags_groups = []
        keys = list(tags_list.keys())
        for values in zip(*(tags_list[key] for key in keys)):
            tags = {key: value for key, value in zip(keys, values)}
            tags_groups.append(tags)

        return tags_groups

    @classmethod
    def data_mat(
        cls, attr: str, subjects: list["SubjectRegions"], selected_regions: list[CountedRegionNode],
        tags_groups: list[dict], summary_func = np.mean,
    ) -> np.ndarray:
        data = np.empty((len(selected_regions), len(tags_groups)))
        for i, tags in enumerate(tags_groups):
            root = cls.summarize_subjects(subjects, summary_func, **tags)
            for j, region in enumerate(selected_regions):
                data[j, i] = getattr(root.locate_node(region), attr)

        return data

    @classmethod
    def show_data_mat(
            cls, data: np.ndarray, selected_regions: list[CountedRegionNode], tags_groups: list[dict],
            tag_names: list[str], title: str = "",
    ):
        fig, ax = plt.subplots()
        im = ax.imshow(data)
        cbar = ax.figure.colorbar(im, ax=ax)

        tag_strs = [" - ".join(tags[n] for n in tag_names) for tags in tags_groups]
        ax.set_xticks(range(len(tag_strs)), labels=tag_strs,
                      rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(selected_regions)), labels=[node.acronym for node in selected_regions])

        ax.set_title(title)
        ax.set_ylabel("")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    regions_csv = Path(__file__).parent / "UnifiedAtlas_Label_ontology_v2.csv"
    count_root = Path(__file__).parent / "counts"
    subjects_metadata_csv = Path(__file__).parent / "cellcount_metadata.csv"

    root_region = CountedRegionNode.load_region_graph_from_ontology(regions_csv)

    subjects_metadata = SubjectRegions.parse_subject_metadata(subjects_metadata_csv)
    subjects = SubjectRegions.parse_directory(
        count_root, root_region, exclude_regions={2285, 2278, 728, 997}, subjects_metadata=subjects_metadata,
    )

    selected_regions = root_region.get_regions(acronyms=("MOB",), include_roots=True)
    tags_groups = SubjectRegions.build_tags_zip(
        condition=["TMT", "TMT", "Blank"] * 2, strain=["TRAP1wt", "TRAP1", "TRAP1"] * 2,
        sex=["F", "F", "F", "M", "M", "M"], cell_label=["IHC"] * 6,
    )
    img = SubjectRegions.data_mat("cell_density", subjects, selected_regions, tags_groups)
    SubjectRegions.show_data_mat(
        img, selected_regions, tags_groups, tag_names=["condition", "strain"], title="Density (cells / mm3)"
    )

    # summarized_root = SubjectRegions.summarize_subjects(subjects, condition="TMT", sex="M")
    # regions = summarized_root.get_regions(acronyms=("MOB", ), include_roots=True)
    # pprint(list(regions))
