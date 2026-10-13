from pathlib import Path

from rag_eval.commands.clean import dev_run_dirs, dev_summary_files


def test_clean_targets_dev_only(tmp_path: Path):
    for rel in ("base/dev/w0.6", "base/dev/w0.7", "chunk1500/dev/w0.6", "chunk1500/heldout/w0.6"):
        (tmp_path / rel).mkdir(parents=True)
        (tmp_path / rel / "summary.csv").write_text("x", encoding="utf-8")
    for rel in ("compare_dev.md", "answer_hit_dev.md", "ci_dev.md", "ci_dev_base.md", "answer_hit_heldout.md", "base/dev/sweep.md"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x", encoding="utf-8")
    dirs = dev_run_dirs(tmp_path)
    assert all("heldout" not in d.parts for d in dirs) and len(dirs) == 3
    files = {f.name for f in dev_summary_files(tmp_path)}
    assert files == {"compare_dev.md", "answer_hit_dev.md", "ci_dev.md", "ci_dev_base.md", "sweep.md"}
    assert "answer_hit_heldout.md" not in files
