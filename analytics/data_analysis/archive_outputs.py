import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive generated outputs into a timestamped folder."
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("archive"),
        help="Root folder where timestamped archive folders are created.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without moving files.",
    )
    return parser.parse_args()


def _collect_output_paths() -> list[Path]:
    legacy_patterns = [
        "analytics/data_analysis/simulated_dataset_train.csv",
        "analytics/data_analysis/simulated_dataset_test.csv",
        "analytics/data_analysis/simulated_dataset_train_woe.csv",
        "analytics/data_analysis/simulated_dataset_test_woe.csv",
        "analytics/data_analysis/variable_selection_report.csv",
        "analytics/data_analysis/woe_mappings.json",
        "analytics/data_analysis/model_comparison.csv",
    ]

    paths = [
        Path("outputs") / "simulator",
        Path("analytics") / "data_analysis" / "artifacts",
        Path("analytics") / "data_analysis" / "models",
        Path("analytics") / "data_analysis" / "model_comparison",
        Path("analytics") / "data_analysis" / "outputs",
        Path("analytics") / "data_analysis" / "__pycache__",
    ]
    paths.extend(Path(p) for p in legacy_patterns)

    existing = [p for p in paths if p.exists()]
    # Sort deeper paths first to avoid trying to move a child after parent moved.
    existing.sort(key=lambda p: len(p.parts), reverse=True)
    return existing


def _move_path(src: Path, archive_dir: Path, dry_run: bool) -> tuple[Path, Path]:
    rel = src
    dest = archive_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        suffix = datetime.now().strftime("%H%M%S")
        dest = dest.with_name(f"{dest.name}_{suffix}")

    if not dry_run:
        shutil.move(str(src), str(dest))
    return src, dest


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = args.archive_root / f"cleanup_{timestamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    to_move = _collect_output_paths()
    moved = []

    for src in to_move:
        src_moved = any(str(src).startswith(str(prev_src) + "\\") for prev_src, _ in moved)
        if src_moved:
            continue
        moved.append(_move_path(src, archive_dir, args.dry_run))

    manifest = {
        "timestamp": timestamp,
        "dry_run": args.dry_run,
        "archive_dir": str(archive_dir),
        "items": [{"from": str(src), "to": str(dest)} for src, dest in moved],
    }
    manifest_path = archive_dir / "archive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Archive folder: {archive_dir}")
    print(f"Items {'planned' if args.dry_run else 'moved'}: {len(moved)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
