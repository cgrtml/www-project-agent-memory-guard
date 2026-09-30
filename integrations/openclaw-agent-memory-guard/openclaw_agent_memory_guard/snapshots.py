"""Snapshots, diff and rollback for a memory workspace.

AMG's ``SnapshotStore`` is an in-process ring buffer keyed by memory keys.
A workspace is a set of files on disk that must survive process restarts, so
this module stores snapshots as JSON files under ``<workspace>/.amg/snapshots``
and restores them by rewriting the memory files.

A snapshot records every memory file's content and a SHA-256 digest of the
whole set. The digest is recomputed and compared on rollback, so a snapshot
edited by hand or by another process is refused instead of restored.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openclaw_agent_memory_guard.entries import Entry, split_hermes, split_openclaw
from openclaw_agent_memory_guard.layouts import Layout, MemoryFile

SNAPSHOT_DIR = Path(".amg") / "snapshots"


@dataclass(frozen=True)
class WorkspaceSnapshot:
    snapshot_id: str
    timestamp: float
    label: str
    layout: Layout
    files: dict[str, str]  # workspace-relative path -> content
    digest: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "label": self.label,
            "layout": self.layout.value,
            "files": self.files,
            "digest": self.digest,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkspaceSnapshot:
        return cls(
            snapshot_id=str(data["snapshot_id"]),
            timestamp=float(data["timestamp"]),
            label=str(data.get("label", "")),
            layout=Layout(data.get("layout", Layout.OPENCLAW.value)),
            files={str(k): str(v) for k, v in data["files"].items()},
            digest=str(data["digest"]),
            metadata=dict(data.get("metadata", {})),
        )


def digest_files(files: dict[str, str]) -> str:
    h = hashlib.sha256()
    for path in sorted(files):
        h.update(path.encode("utf-8"))
        h.update(b"\0")
        h.update(files[path].encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


@dataclass(frozen=True)
class EntryChange:
    kind: str  # "added" | "removed" | "changed"
    key: str
    file: str
    line: int
    before: str | None
    after: str | None


class WorkspaceSnapshotStore:
    """File-backed snapshot store for one workspace root."""

    def __init__(self, root: Path, *, directory: Path | None = None) -> None:
        self.root = Path(root)
        self.directory = directory if directory is not None else self.root / SNAPSHOT_DIR

    # ---- capture ---------------------------------------------------------

    def capture(
        self,
        files: list[MemoryFile],
        *,
        layout: Layout,
        label: str = "manual",
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot:
        contents = {
            mf.relative_path: mf.path.read_text(encoding="utf-8", errors="replace") for mf in files
        }
        snapshot = WorkspaceSnapshot(
            snapshot_id=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8],
            timestamp=time.time(),
            label=label,
            layout=layout,
            files=contents,
            digest=digest_files(contents),
            metadata=dict(metadata or {}),
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{snapshot.snapshot_id}.json"
        path.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")
        return snapshot

    # ---- read ------------------------------------------------------------

    def list(self) -> list[WorkspaceSnapshot]:
        if not self.directory.is_dir():
            return []
        found = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                found.append(
                    WorkspaceSnapshot.from_dict(json.loads(path.read_text(encoding="utf-8")))
                )
            except (ValueError, KeyError):
                continue
        return found

    def get(self, snapshot_id: str) -> WorkspaceSnapshot | None:
        for snap in self.list():
            if snap.snapshot_id == snapshot_id:
                return snap
        return None

    def latest(self) -> WorkspaceSnapshot | None:
        snaps = self.list()
        return snaps[-1] if snaps else None

    # ---- rollback --------------------------------------------------------

    def rollback(self, snapshot_id: str | None = None) -> WorkspaceSnapshot:
        """Rewrite the memory files from a snapshot. Files present in the
        workspace but absent from the snapshot are left alone; nothing is
        deleted. Refuses a snapshot whose digest no longer matches its files."""
        snap = self.get(snapshot_id) if snapshot_id else self.latest()
        if snap is None:
            raise FileNotFoundError("no snapshot to roll back to")
        if digest_files(snap.files) != snap.digest:
            raise ValueError(f"snapshot {snap.snapshot_id} failed its digest check; not restoring")
        for rel, content in snap.files.items():
            target = (self.root / rel).resolve()
            if self.root.resolve() not in target.parents:
                raise ValueError(f"snapshot path escapes the workspace: {rel}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return snap


def _entries(layout: Layout, files: dict[str, str]) -> dict[str, Entry]:
    out: dict[str, Entry] = {}
    for rel, content in files.items():
        split = split_hermes if layout == Layout.HERMES else split_openclaw
        for entry in split(rel, content):
            out[entry.key] = entry
    return out


def diff_entries(
    before: dict[str, str], after: dict[str, str], *, layout: Layout
) -> list[EntryChange]:
    """Entry-level diff between two file sets.

    Tracked OpenClaw entries (promotion markers) compare by key. Untracked
    blocks are keyed by line number, so a block that moved shows as removed
    and added; comparing by text collapses pure moves into no change.
    """
    old = _entries(layout, before)
    new = _entries(layout, after)
    old_texts = {e.text for e in old.values()}
    new_texts = {e.text for e in new.values()}
    changes: list[EntryChange] = []
    for key, entry in new.items():
        if entry.tracked:
            if key not in old:
                changes.append(
                    EntryChange("added", key, _file_of(key), entry.line, None, entry.text)
                )
            elif old[key].text != entry.text:
                changes.append(
                    EntryChange(
                        "changed", key, _file_of(key), entry.line, old[key].text, entry.text
                    )
                )
        elif entry.text not in old_texts:
            changes.append(EntryChange("added", key, _file_of(key), entry.line, None, entry.text))
    for key, entry in old.items():
        gone = key not in new if entry.tracked else entry.text not in new_texts
        if gone:
            changes.append(EntryChange("removed", key, _file_of(key), entry.line, entry.text, None))
    return changes


def _file_of(key: str) -> str:
    return key.rsplit("#", 1)[0]
