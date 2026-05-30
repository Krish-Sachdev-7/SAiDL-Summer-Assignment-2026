"""Small helper for run notes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv
import re
from typing import Any


@dataclass
class DocArtifacts:
    markdown_path: Path
    latex_path: Path
    csv_path: Path
    include_index_path: Path


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9_-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "run"


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            next_key = f"{prefix}.{k}" if prefix else str(k)
            _flatten(next_key, v, out)
        return

    if isinstance(value, (list, tuple)):
        out[prefix] = "[" + ", ".join(str(x) for x in value) + "]"
        return

    out[prefix] = value


def _ensure_report_scaffold(track_report_dir: Path) -> None:
    auto_dir = track_report_dir / "auto"
    auto_dir.mkdir(parents=True, exist_ok=True)
    (auto_dir / "auto_experiments.tex").touch(exist_ok=True)


def update_experiment_docs(
    track: str,
    experiment_name: str,
    cfg_dict: dict[str, Any],
    metrics: dict[str, Any],
    run_output_dir: str,
    repo_root: Path,
) -> DocArtifacts:
    """Write run-note artifacts."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_slug = _slugify(f"{experiment_name}_{timestamp}")

    track_report_dir = repo_root / track / "report"
    _ensure_report_scaffold(track_report_dir)
    auto_dir = track_report_dir / "auto"

    md_path = auto_dir / f"run_{run_slug}.md"
    tex_path = auto_dir / f"run_{run_slug}.tex"
    csv_path = auto_dir / "runs.csv"
    include_index_path = auto_dir / "auto_experiments.tex"

    flat_cfg: dict[str, Any] = {}
    _flatten("", cfg_dict, flat_cfg)

    md_lines = [
        f"# Run {experiment_name}",
        "",
        f"- track: {track}",
        f"- timestamp: {timestamp}",
        f"- hydra_output_dir: {run_output_dir}",
        "",
        "## Final Metrics",
    ]
    for k in sorted(metrics):
        md_lines.append(f"- {k}: {metrics[k]}")

    md_lines += ["", "## Resolved Config"]
    for k in sorted(flat_cfg):
        md_lines.append(f"- {k}: {flat_cfg[k]}")

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    metric_lines = ", ".join(
        f"{k.replace('_', '\\_')}={metrics[k]}" for k in sorted(metrics)
    )
    tex_lines = [
        "% Auto-generated experiment snippet",
        f"\\subsection{{Run: {experiment_name.replace('_', '\\_')}}}",
        f"\\textbf{{Output directory.}} {run_output_dir.replace('_', '\\_')}.",
        "",
        f"\\textbf{{Final metrics.}} {metric_lines}.",
        "",
        "\\paragraph{Configuration.}",
    ]
    for k in sorted(flat_cfg):
        tex_lines.append(f"{k.replace('_', '\\_')}: {str(flat_cfg[k]).replace('_', '\\_')}\\\\")

    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    row = {
        "timestamp": timestamp,
        "track": track,
        "experiment_name": experiment_name,
        "run_output_dir": run_output_dir,
        **{k: metrics[k] for k in sorted(metrics.keys())},
    }
    base_fields = ["timestamp", "track", "experiment_name", "run_output_dir"]
    rows: list[dict[str, Any]] = []
    fieldnames = [*base_fields, *sorted(metrics.keys())]
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            existing_fields = list(reader.fieldnames or [])
        extra_fields = sorted(
            set(existing_fields)
            .union(row.keys())
            .difference(base_fields)
        )
        fieldnames = [*base_fields, *extra_fields]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for existing_row in rows:
            writer.writerow(existing_row)
        writer.writerow(row)

    snippet_paths = sorted(auto_dir.glob("run_*.tex"))
    include_lines = [
        "% Auto-generated. Included by report root file.",
        "% Newest run appears first.",
    ]
    for p in reversed(snippet_paths):
        include_lines.append(f"\\input{{auto/{p.name}}}")
        include_lines.append("")
    include_index_path.write_text("\n".join(include_lines) + "\n", encoding="utf-8")

    return DocArtifacts(
        markdown_path=md_path,
        latex_path=tex_path,
        csv_path=csv_path,
        include_index_path=include_index_path,
    )
