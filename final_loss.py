#!/usr/bin/env python3
"""Summarize final train/eval loss for the latest cramming wave runs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


WAVES = ("sinusoid", "square", "triangular", "sawtooth")


@dataclass
class WaveResult:
    wave: str
    run_dir: Path
    csv_path: Path
    final_train_loss: float
    final_eval_loss: float
    train_diff_pct_vs_sinusoid: float
    eval_diff_pct_vs_sinusoid: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-dir",
        default=Path(__file__).resolve().parent / "outputs",
        type=Path,
        help="Base cramming outputs directory.",
    )
    return parser.parse_args()


def latest_csv_for_wave(outputs_dir: Path, wave: str) -> tuple[Path, Path]:
    wave_dir = outputs_dir / f"cramming_{wave}_single"
    candidates = sorted(wave_dir.glob("*/table_*_convergence_results.csv"))
    if not candidates:
        raise FileNotFoundError(f"No convergence CSV found for wave '{wave}' in {wave_dir}.")

    csv_path = candidates[-1]
    return csv_path.parent, csv_path


def read_last_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"CSV has no data rows: {csv_path}")
    return rows[-1]


def compute_results(outputs_dir: Path) -> list[WaveResult]:
    raw_rows: dict[str, tuple[Path, Path, dict[str, str]]] = {}
    for wave in WAVES:
        run_dir, csv_path = latest_csv_for_wave(outputs_dir, wave)
        raw_rows[wave] = (run_dir, csv_path, read_last_row(csv_path))

    sinusoid_train = float(raw_rows["sinusoid"][2]["loss"])
    sinusoid_eval = float(raw_rows["sinusoid"][2]["eval_loss"])

    results: list[WaveResult] = []
    for wave in WAVES:
        run_dir, csv_path, last_row = raw_rows[wave]
        train_loss = float(last_row["loss"])
        eval_loss = float(last_row["eval_loss"])
        results.append(
            WaveResult(
                wave=wave,
                run_dir=run_dir,
                csv_path=csv_path,
                final_train_loss=train_loss,
                final_eval_loss=eval_loss,
                train_diff_pct_vs_sinusoid=((train_loss - sinusoid_train) / sinusoid_train) * 100.0,
                eval_diff_pct_vs_sinusoid=((eval_loss - sinusoid_eval) / sinusoid_eval) * 100.0,
            )
        )
    return results


def format_table(results: list[WaveResult]) -> str:
    header = [
        "wave",
        "run_dir",
        "final_train_loss",
        "final_eval_loss",
        "train_diff_vs_sinusoid_pct",
        "eval_diff_vs_sinusoid_pct",
    ]
    rows = [header]
    for result in results:
        rows.append(
            [
                result.wave,
                result.run_dir.name,
                f"{result.final_train_loss:.6f}",
                f"{result.final_eval_loss:.6f}",
                f"{result.train_diff_pct_vs_sinusoid:+.4f}",
                f"{result.eval_diff_pct_vs_sinusoid:+.4f}",
            ]
        )

    widths = [max(len(row[idx]) for row in rows) for idx in range(len(header))]
    return "\n".join("  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) for row in rows)


def main() -> int:
    args = parse_args()
    results = compute_results(args.outputs_dir.resolve())
    print(format_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
