"""Batch analysis of all RAW files in a target folder.

Usage:
    python batch_analyze.py <targets_folder> [--results-dir <dir>]

For each RAW file found, runs `cc.py analyze` and writes a consolidated
deltae_report.csv in the results directory.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

RAW_EXTENSIONS = {".nef", ".cr2", ".cr3", ".arw", ".raf", ".dng", ".tif", ".tiff"}

FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def find_raw_files(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in RAW_EXTENSIONS
    )


def _first_float(text: str) -> float | None:
    m = FLOAT_RE.search(text)
    return float(m.group()) if m else None


def parse_stdout(text: str) -> dict[str, object]:
    info: dict[str, object] = {}

    for line in text.splitlines():
        # orientation: rotation_steps=0 mirrored=True white_patch=19 black_patch=24
        if line.startswith("orientation:"):
            m = re.search(r"rotation_steps=(\d+)", line)
            if m:
                info["orientation_steps"] = int(m.group(1))
            m = re.search(r"mirrored=(True|False)", line)
            if m:
                info["orientation_mirrored"] = m.group(1) == "True"
            m = re.search(r"white_patch=(\d+)", line)
            if m:
                info["white_patch"] = int(m.group(1))
            m = re.search(r"black_patch=(\d+)", line)
            if m:
                info["black_patch"] = int(m.group(1))

        # used (camera-wb):   [...]  ~N K
        m = re.match(r"\s+used \((.+?)\):", line)
        if m:
            info["scene_white_source"] = m.group(1)

        # Selected k: 3
        m = re.match(r"Selected k:\s*(\d+)", line)
        if m:
            info["selected_k"] = int(m.group(1))

    # Parse the three deltaE00 blocks (mean/median/max) for each model.
    # Blocks appear in order: Baseline, RPCC, HPPCC, HPPCC+RPCC.
    block_patterns = [
        ("baseline", re.compile(r"^Baseline white-preserving")),
        ("rpcc",     re.compile(r"^Global RPCC")),
        ("hppcc",    re.compile(r"^HPPCC \(")),
        ("hppcc_rpcc", re.compile(r"^HPPCC \+ RPCC residual")),
    ]
    lines = text.splitlines()
    for label, header_re in block_patterns:
        for i, line in enumerate(lines):
            if header_re.match(line.strip()):
                block = "\n".join(lines[i: i + 10])
                m_mean   = re.search(r"deltaE00 mean:\s*(" + FLOAT_RE.pattern + r")", block)
                m_median = re.search(r"deltaE00 median:\s*(" + FLOAT_RE.pattern + r")", block)
                m_max    = re.search(r"deltaE00 max:\s*(" + FLOAT_RE.pattern + r")", block)
                if m_mean:
                    info[f"{label}_mean"]   = float(m_mean.group(1))
                    info[f"{label}_median"] = float(m_median.group(1)) if m_median else None
                    info[f"{label}_max"]    = float(m_max.group(1))    if m_max    else None
                break

    return info


def analyze_one(
    raw_path: Path,
    output_dir: Path,
    python_exe: Path,
    app: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python_exe), str(app),
        "analyze",
        "--cc-image", str(raw_path),
        "--analysis-dir", str(output_dir),
        "--process-dir", str(output_dir),
        "--no-show-detection-preview",
        "--no-show-developed-image-preview",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout = result.stdout
    stderr = result.stderr
    (output_dir / "analysis_stdout.txt").write_text(stdout + stderr, encoding="utf-8")

    row: dict[str, object] = {"file": raw_path.name, "output_dir": str(output_dir)}

    if result.returncode != 0:
        # Extract last meaningful error line
        error_lines = [ln.strip() for ln in (stderr + stdout).splitlines() if ln.strip()]
        error_msg = error_lines[-1] if error_lines else "unknown error"
        # Strip leading exception class
        error_msg = re.sub(r"^\w+(?:\.\w+)*Error:\s*", "", error_msg)
        error_msg = re.sub(r"^\w+(?:\.\w+)*Exception:\s*", "", error_msg)
        row["status"] = "error"
        row["error"] = error_msg
        overlay = output_dir / "checker_detection_overlay.png"
        row["has_overlay"] = overlay.exists()
        return row

    parsed = parse_stdout(stdout)
    row["status"] = "ok"
    row["error"] = ""
    row.update(parsed)
    overlay = output_dir / "checker_detection_overlay.png"
    row["has_overlay"] = overlay.exists()
    return row


CSV_FIELDS = [
    "file", "status", "error",
    "selected_k", "scene_white_source",
    "orientation_steps", "orientation_mirrored",
    "white_patch", "black_patch",
    "baseline_mean", "baseline_median", "baseline_max",
    "rpcc_mean", "rpcc_median", "rpcc_max",
    "hppcc_mean", "hppcc_median", "hppcc_max",
    "hppcc_rpcc_mean", "hppcc_rpcc_median", "hppcc_rpcc_max",
    "has_overlay", "output_dir",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch HPPCC analysis of a RAW folder.")
    parser.add_argument("targets_folder", type=Path, help="Folder containing RAW files.")
    parser.add_argument(
        "--results-dir", type=Path, default=None,
        help="Output root. Default: <targets_folder>/Results",
    )
    args = parser.parse_args()

    targets_folder: Path = args.targets_folder.resolve()
    results_dir: Path = (args.results_dir or targets_folder / "Results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parent
    python_exe = project_root / ".venv" / "Scripts" / "python.exe"
    app = project_root / "src" / "cc.py"

    raw_files = find_raw_files(targets_folder)
    if not raw_files:
        print(f"No RAW files found in {targets_folder}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(raw_files)} RAW file(s). Results -> {results_dir}")

    rows = []
    for raw_path in raw_files:
        output_dir = results_dir / raw_path.stem
        print(f"  Analyzing {raw_path.name} ...", end=" ", flush=True)
        row = analyze_one(raw_path, output_dir, python_exe, app)
        status = row.get("status", "?")
        if status == "ok":
            de = row.get("hppcc_rpcc_mean") or row.get("hppcc_mean")
            print(f"ok  hppcc+rpcc mean dE={de:.2f}" if de is not None else "ok")
        else:
            print(f"ERROR: {row.get('error', '')}")
        rows.append(row)

    csv_path = results_dir / "deltae_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nReport written to: {csv_path}")

    ok = sum(1 for r in rows if r.get("status") == "ok")
    errors = len(rows) - ok
    print(f"Summary: {ok} ok, {errors} errors out of {len(rows)} files.")


if __name__ == "__main__":
    main()
