"""Data access helpers for the interactive volume inspector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from apps.common import volume_utils as vu


@dataclass(frozen=True)
class DataRoot:
    """Container describing a dataset root used in the inspector."""

    name: str
    path: Path

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").title()


@dataclass(frozen=True)
class CaseDescriptor:
    """Descriptor for an individual BraTS case."""

    key: str
    root_name: str
    case_id: str
    path: Path

    def to_option(self) -> Mapping[str, str]:
        return {"label": self.case_id, "value": self.key}


class VolumeRepository:
    """Repository abstraction over on-disk BraTS cases."""

    def __init__(self, data_roots: Sequence[Path]) -> None:
        resolved = [Path(root).expanduser().resolve() for root in data_roots]
        missing = [root for root in resolved if not root.is_dir()]
        if len(missing) == len(resolved):
            raise FileNotFoundError("None of the provided data roots exist.")
        self._roots: List[DataRoot] = [DataRoot(name=root.name, path=root) for root in resolved if root.is_dir()]
        self._cases_by_root: Dict[str, List[CaseDescriptor]] = {}
        self._case_lookup: Dict[str, CaseDescriptor] = {}
        self._build_index()

    # ------------------------------------------------------------------
    # Index construction
    def _build_index(self) -> None:
        for root in self._roots:
            cases: List[CaseDescriptor] = []
            for case_dir in vu.discover_cases([root.path]):
                case_id = case_dir.name
                key = self._make_key(root.name, case_id)
                descriptor = CaseDescriptor(key=key, root_name=root.name, case_id=case_id, path=case_dir)
                cases.append(descriptor)
                self._case_lookup[key] = descriptor
            cases.sort(key=lambda c: c.case_id)
            self._cases_by_root[root.name] = cases

    @staticmethod
    def _make_key(root_name: str, case_id: str) -> str:
        return f"{root_name}|{case_id}"

    # ------------------------------------------------------------------
    # Public API
    @property
    def roots(self) -> List[DataRoot]:
        return list(self._roots)

    def dataset_options(self) -> List[Mapping[str, str]]:
        return [{"label": root.label, "value": root.name} for root in self._roots if self._cases_by_root.get(root.name)]

    def default_root(self) -> str | None:
        options = self.dataset_options()
        return options[0]["value"] if options else None

    def case_options(self, root_name: str) -> List[Mapping[str, str]]:
        cases = self._cases_by_root.get(root_name, [])
        return [descriptor.to_option() for descriptor in cases]

    def default_case_key(self, root_name: str | None = None) -> str | None:
        if root_name is None:
            root_name = self.default_root()
        if root_name is None:
            return None
        options = self.case_options(root_name)
        return options[0]["value"] if options else None

    def resolve(self, case_key: str | None) -> CaseDescriptor | None:
        if case_key is None:
            return None
        return self._case_lookup.get(case_key)

    # Metadata and volume access ------------------------------------------------
    def case_metadata(self, case_key: str) -> MutableMapping[str, object]:
        descriptor = self._case_lookup[case_key]
        meta = vu.case_metadata(descriptor.path)
        meta["root_name"] = descriptor.root_name
        meta["case_key"] = case_key
        return meta

    def volumes(self, case_key: str) -> Tuple[Dict[str, np.ndarray], np.ndarray | None]:
        descriptor = self._case_lookup[case_key]
        return vu.cached_volumes(descriptor.path)


__all__ = ["DataRoot", "CaseDescriptor", "VolumeRepository"]
