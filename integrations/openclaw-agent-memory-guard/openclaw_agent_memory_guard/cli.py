"""``amg-openclaw``: scan OpenClaw or Hermes memory files with AMG."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from agent_memory_guard import Policy
from agent_memory_guard.policies.policy import load_policy
from agent_memory_guard.scanner import format_sarif, format_text
from openclaw_agent_memory_guard.layouts import Layout, discover_hermes, discover_openclaw
from openclaw_agent_memory_guard.scan import scan_workspace
from openclaw_agent_memory_guard.snapshots import WorkspaceSnapshotStore, diff_entries


def _load_policy(name: str) -> Policy:
    if name == "strict":
        return Policy.strict()
    if name == "tiered":
        return Policy.tiered()
    if name == "permissive":
        return Policy.permissive()
    path = Path(name)
    if path.suffix in (".yml", ".yaml") and path.is_file():
        return load_policy(path)
    raise SystemExit(f"Unknown policy: {name} (use strict, tiered, permissive or a YAML file)")


def cmd_scan(args: argparse.Namespace) -> int:
    root = Path(args.workspace).expanduser()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        return 2
    overrides = None
    if args.provenance:
        overrides = json.loads(Path(args.provenance).read_text(encoding="utf-8"))
    result = scan_workspace(
        root,
        layout=Layout(args.layout),
        policy=_load_policy(args.policy),
        provenance_overrides=overrides,
    )
    if args.format == "json":
        output = result.to_json()
    elif args.format == "sarif":
        output = format_sarif(result.to_scan_result())
    else:
        output = format_text(result.to_scan_result())
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(output)
    if args.snapshot:
        store = WorkspaceSnapshotStore(root)
        snap = store.capture(
            _files_for(root, args),
            layout=Layout(args.layout),
            label="scan",
            metadata={"flagged": len(result.flagged), "entries": len(result.verdicts)},
        )
        print(f"Snapshot {snap.snapshot_id} saved under {store.directory}", file=sys.stderr)
    if args.fail_on_findings and result.flagged:
        return 1
    return 0


def _files_for(root: Path, args: argparse.Namespace):
    if Layout(args.layout) == Layout.HERMES:
        return discover_hermes(root)
    return discover_openclaw(root)


def _root(args: argparse.Namespace) -> Path | None:
    root = Path(args.workspace).expanduser()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        return None
    return root


def cmd_snapshot(args: argparse.Namespace) -> int:
    root = _root(args)
    if root is None:
        return 2
    store = WorkspaceSnapshotStore(root)
    snap = store.capture(_files_for(root, args), layout=Layout(args.layout), label=args.label)
    print(f"{snap.snapshot_id}  {len(snap.files)} files  {snap.digest[:12]}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    root = _root(args)
    if root is None:
        return 2
    snaps = WorkspaceSnapshotStore(root).list()
    if not snaps:
        print("No snapshots.")
        return 0
    for snap in snaps:
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(snap.timestamp))
        print(f"{snap.snapshot_id}  {when}  {snap.label:<8}  {len(snap.files)} files")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    """Entry-level diff between a snapshot and the working files (or two snapshots)."""
    root = _root(args)
    if root is None:
        return 2
    store = WorkspaceSnapshotStore(root)
    base = store.get(args.base) if args.base else store.latest()
    if base is None:
        print("Error: no base snapshot", file=sys.stderr)
        return 2
    if args.target:
        target = store.get(args.target)
        if target is None:
            print(f"Error: snapshot {args.target} not found", file=sys.stderr)
            return 2
        after = target.files
    else:
        after = {
            mf.relative_path: mf.path.read_text(encoding="utf-8", errors="replace")
            for mf in _files_for(root, args)
        }
    layout = Layout(args.layout)
    changes = diff_entries(base.files, after, layout=layout)
    if not changes:
        print(f"No entry changes since {base.snapshot_id}.")
        return 0
    verdicts = {}
    if not args.no_scan and not args.target:
        result = scan_workspace(root, layout=layout, policy=_load_policy(args.policy))
        verdicts = {v.entry.key: v for v in result.verdicts}
    for change in changes:
        verdict = verdicts.get(change.key)
        tag = (
            f"  [{verdict.action.value}]"
            if verdict is not None and change.kind != "removed"
            else ""
        )
        print(f"{change.kind:<8} {change.key}{tag}")
        shown = change.after if change.after is not None else change.before
        for line in (shown or "").splitlines()[:6]:
            print(f"    {line}")
    return 1 if args.fail_on_changes else 0


def cmd_rollback(args: argparse.Namespace) -> int:
    root = _root(args)
    if root is None:
        return 2
    store = WorkspaceSnapshotStore(root)
    try:
        snap = store.rollback(args.to)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Restored {len(snap.files)} files from {snap.snapshot_id} ({snap.label}).")
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace",
        default="~/.openclaw/workspace",
        help="OpenClaw workspace or Hermes memories directory",
    )
    parser.add_argument(
        "--layout", choices=[item.value for item in Layout], default=Layout.OPENCLAW.value
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amg-openclaw",
        description="Scan OpenClaw or Hermes Agent memory files with OWASP Agent Memory Guard.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="Scan a memory workspace and report per-entry verdicts")
    _add_common(scan)
    scan.add_argument(
        "--snapshot", action="store_true", help="Save a snapshot of the memory files after the scan"
    )
    scan.add_argument(
        "--policy", default="strict", help="strict, tiered, permissive or a policy YAML file"
    )
    scan.add_argument(
        "--provenance", help="JSON file mapping workspace-relative path to OpenClaw origin class"
    )
    scan.add_argument("--format", choices=["text", "json", "sarif"], default="text")
    scan.add_argument("--output", help="Write the report to a file instead of stdout")
    scan.add_argument(
        "--fail-on-findings", action="store_true", help="Exit 1 when any entry is flagged"
    )
    scan.set_defaults(func=cmd_scan)

    snapshot = sub.add_parser("snapshot", help="Save a snapshot of the memory files")
    _add_common(snapshot)
    snapshot.add_argument("--label", default="manual")
    snapshot.set_defaults(func=cmd_snapshot)

    listing = sub.add_parser("list", help="List saved snapshots")
    _add_common(listing)
    listing.set_defaults(func=cmd_list)

    diff = sub.add_parser(
        "diff", help="Show entries added, removed or changed since a snapshot, with verdicts"
    )
    _add_common(diff)
    diff.add_argument("--base", help="Snapshot id to compare from (default: latest)")
    diff.add_argument("--target", help="Snapshot id to compare to (default: working files)")
    diff.add_argument("--policy", default="strict")
    diff.add_argument("--no-scan", action="store_true", help="Skip detector verdicts")
    diff.add_argument(
        "--fail-on-changes", action="store_true", help="Exit 1 when any entry changed"
    )
    diff.set_defaults(func=cmd_diff)

    rollback = sub.add_parser("rollback", help="Restore memory files from a snapshot")
    _add_common(rollback)
    rollback.add_argument("--to", help="Snapshot id (default: latest)")
    rollback.set_defaults(func=cmd_rollback)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
