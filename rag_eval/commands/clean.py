"""`clean`: delete dev run outputs before a re-run. Heldout results are never touched."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def dev_run_dirs(results_dir: Path) -> list[Path]:
    return sorted(d for d in results_dir.glob("*/dev/*") if d.is_dir())


def dev_summary_files(results_dir: Path) -> list[Path]:
    names = ("compare_dev.md", "answer_hit_dev.md")
    files = [results_dir / n for n in names if (results_dir / n).exists()]
    files += sorted(results_dir.glob("ci_dev*.md"))
    files += sorted(results_dir.glob("*/dev/sweep*.md"))
    return files


def run(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)
    dirs, files = dev_run_dirs(results_dir), dev_summary_files(results_dir)
    for d in dirs:
        assert "heldout" not in d.parts, d          # belt and braces
        shutil.rmtree(d)
    for f in files:
        f.unlink()
    kept = sorted(p for p in results_dir.glob("*/heldout/*") if p.is_dir())
    print(f"removed {len(dirs)} dev run folder(s) and {len(files)} dev summary file(s); kept {len(kept)} heldout run(s)")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("clean", help="remove dev results only (heldout is never deleted)")
    p.add_argument("--results-dir", default="results")
    p.set_defaults(func=run)
