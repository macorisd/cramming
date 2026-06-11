#!/usr/bin/env python3
"""Plot averaged k-fold curves for cramming positional-wave experiments."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_MPLCONFIGDIR = Path("/tmp") / "cramming_matplotlib"
DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
configured_mpl_dir = os.environ.get("MPLCONFIGDIR")
if not configured_mpl_dir or not os.access(configured_mpl_dir, os.W_OK):
    os.environ["MPLCONFIGDIR"] = str(DEFAULT_MPLCONFIGDIR)

import matplotlib.pyplot as plt


# Embed TrueType fonts in PDFs for better portability.
plt.rcParams["pdf.fonttype"] = "truetype"


REPO_ROOT = Path(__file__).resolve().parent
OUTPUTS_KFOLD_DIR = REPO_ROOT / "outputs-kfold"
GRAPHS_KFOLD_DIR = REPO_ROOT / "graphs_k_fold"
WAVES = ("sinusoid", "triangular", "square", "sawtooth")
WAVE_COLORS = {
    "sinusoid": "r",
    "triangular": "g",
    "square": "b",
    "sawtooth": "orange",
}
RUN_DIR_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"


@dataclass(frozen=True)
class Curve:
    steps: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    folds: int

    def from_step(self, threshold: int) -> Curve | None:
        mask = self.steps >= threshold
        if not mask.any():
            return None
        return Curve(
            steps=self.steps[mask],
            mean=self.mean[mask],
            std=self.std[mask],
            folds=self.folds,
        )


@dataclass(frozen=True)
class WaveData:
    wave: str
    group_dir: Path
    fold_tables: tuple[Path, ...]
    loss: Curve
    eval_loss: Curve | None
    best_eval_loss: Curve | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot averaged k-fold curves for cramming runs.")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=OUTPUTS_KFOLD_DIR,
        help="Base directory containing cramming_*_kfold result folders.",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=GRAPHS_KFOLD_DIR,
        help="Base directory where timestamped graph folders will be created.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Use a specific run-id prefix, e.g. 2026-04-29_22-14-58. Defaults to latest per wave.",
    )
    parser.add_argument(
        "--waves",
        default=",".join(WAVES),
        help="Comma-separated waves to plot. Defaults to all four waves.",
    )
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Do not save one averaged graph per wave.",
    )
    parser.add_argument(
        "--no-compare-sinusoid",
        action="store_true",
        help="Do not save pairwise comparisons against sinusoid.",
    )
    parser.add_argument(
        "--final-step-threshold",
        type=int,
        default=60000,
        help="Step threshold used for additional final-segment graphs.",
    )
    parser.add_argument(
        "--no-final-segment",
        action="store_true",
        help="Do not save additional graphs restricted to the final training segment.",
    )
    return parser.parse_args()


def parse_waves(raw_waves: str) -> tuple[str, ...]:
    waves = tuple(wave.strip() for wave in raw_waves.split(",") if wave.strip())
    invalid = sorted(set(waves) - set(WAVES))
    if invalid:
        raise ValueError(f"Unsupported waves: {', '.join(invalid)}. Supported: {', '.join(WAVES)}.")
    if not waves:
        raise ValueError("At least one wave must be selected.")
    return waves


def parse_group_timestamp(group_dir: Path) -> datetime:
    try:
        return datetime.strptime(group_dir.name[:19], RUN_DIR_TIMESTAMP_FORMAT)
    except ValueError:
        return datetime.min


def latest_group_dir(outputs_dir: Path, wave: str, run_id: str | None) -> Path:
    wave_root = outputs_dir / f"cramming_{wave}_kfold"
    if not wave_root.is_dir():
        raise FileNotFoundError(f"Missing k-fold output directory for wave '{wave}': {wave_root}")

    group_dirs = sorted(path for path in wave_root.iterdir() if path.is_dir())
    if run_id is not None:
        group_dirs = [path for path in group_dirs if path.name.startswith(run_id)]
    if not group_dirs:
        suffix = f" starting with {run_id!r}" if run_id is not None else ""
        raise FileNotFoundError(f"No k-fold run group found for wave '{wave}'{suffix}.")
    return max(group_dirs, key=parse_group_timestamp)


def fold_number(fold_dir: Path) -> int:
    try:
        return int(fold_dir.name.split("_")[-1])
    except ValueError:
        return 0


def find_fold_tables(group_dir: Path, wave: str) -> tuple[Path, ...]:
    tables = []
    for fold_dir in sorted(group_dir.glob("fold_*"), key=fold_number):
        table = fold_dir / f"table_cramming_{wave}_kfold_convergence_results.csv"
        if table.is_file():
            tables.append(table)
    if not tables:
        raise FileNotFoundError(f"No convergence tables found in {group_dir}.")
    return tuple(tables)


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def numeric_value(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def fold_series(rows: list[dict[str, str]], value_key: str, step_key: str) -> dict[float, float]:
    series = {}
    for row in rows:
        step = numeric_value(row, step_key)
        value = numeric_value(row, value_key)
        if step is None or value is None:
            continue
        series[step] = value
    return series


def averaged_curve(fold_rows: Iterable[list[dict[str, str]]], value_key: str, step_key: str) -> Curve | None:
    per_fold = [fold_series(rows, value_key, step_key) for rows in fold_rows]
    per_fold = [series for series in per_fold if series]
    if not per_fold:
        return None

    common_steps = sorted(set.intersection(*(set(series) for series in per_fold)))
    if not common_steps:
        return None

    values = np.asarray([[series[step] for step in common_steps] for series in per_fold], dtype=float)
    return Curve(
        steps=np.asarray(common_steps, dtype=float),
        mean=values.mean(axis=0),
        std=values.std(axis=0),
        folds=len(per_fold),
    )


def load_wave_data(outputs_dir: Path, wave: str, run_id: str | None) -> WaveData:
    group_dir = latest_group_dir(outputs_dir, wave, run_id)
    fold_tables = find_fold_tables(group_dir, wave)
    rows_by_fold = [load_rows(table) for table in fold_tables]

    loss = averaged_curve(rows_by_fold, "loss", "step")
    if loss is None:
        raise ValueError(f"No train loss data found for wave '{wave}' in {group_dir}.")

    return WaveData(
        wave=wave,
        group_dir=group_dir,
        fold_tables=fold_tables,
        loss=loss,
        eval_loss=averaged_curve(rows_by_fold, "eval_loss", "eval_step"),
        best_eval_loss=averaged_curve(rows_by_fold, "best_eval_loss", "eval_step"),
    )


def create_run_graph_dir(graphs_dir: Path) -> Path:
    timestamp = datetime.now().strftime(RUN_DIR_TIMESTAMP_FORMAT)
    run_graph_dir = graphs_dir / timestamp
    run_graph_dir.mkdir(parents=True, exist_ok=False)
    return run_graph_dir


def draw_curve(
    curve: Curve,
    label: str,
    color: str,
    linestyle: str = "-",
    show_std: bool = True,
) -> None:
    plt.plot(curve.steps, curve.mean, color=color, linestyle=linestyle, linewidth=2, label=label)
    if show_std:
        lower = curve.mean - curve.std
        upper = curve.mean + curve.std
        plt.fill_between(curve.steps, lower, upper, color=color, alpha=0.10, linewidth=0)


def select_curve(curve: Curve | None, final_step_threshold: int | None) -> Curve | None:
    if curve is None:
        return None
    if final_step_threshold is None:
        return curve
    return curve.from_step(final_step_threshold)


def finish_plot(save_path: Path, title: str, ylabel: str, log_scale: bool = False) -> None:
    plt.xlabel("Step", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    if log_scale:
        plt.yscale("log")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, which="both", axis="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Graph saved: {save_path}")


def draw_loss_comparison(
    waves: Iterable[WaveData],
    save_path: Path,
    log_scale: bool = False,
    final_step_threshold: int | None = None,
) -> None:
    plt.figure(figsize=(10, 6))
    for wave_data in waves:
        color = WAVE_COLORS[wave_data.wave]
        loss = select_curve(wave_data.loss, final_step_threshold)
        eval_loss = select_curve(wave_data.eval_loss, final_step_threshold)
        if loss is not None:
            draw_curve(loss, f"{wave_data.wave.capitalize()} Train Loss", color, "-")
        if eval_loss is not None:
            draw_curve(eval_loss, f"{wave_data.wave.capitalize()} Eval Loss", color, "--")

    suffix = " (Log Scale)" if log_scale else ""
    title_segment = f" from Step {final_step_threshold}" if final_step_threshold is not None else ""
    finish_plot(
        save_path,
        f"K-Fold Average Train and Eval Loss{title_segment}{suffix}",
        "Loss (log scale)" if log_scale else "Loss",
        log_scale=log_scale,
    )


def draw_metric_comparison(
    waves: Iterable[WaveData],
    metric_attr: str,
    metric_label: str,
    save_path: Path,
    final_step_threshold: int | None = None,
) -> None:
    plt.figure(figsize=(10, 6))
    for wave_data in waves:
        curve = select_curve(getattr(wave_data, metric_attr), final_step_threshold)
        if curve is None:
            continue
        draw_curve(curve, wave_data.wave.capitalize(), WAVE_COLORS[wave_data.wave])

    title_segment = f" from Step {final_step_threshold}" if final_step_threshold is not None else ""
    finish_plot(save_path, f"K-Fold Average {metric_label}{title_segment}", metric_label)


def draw_individual_wave(wave_data: WaveData, save_dir: Path, final_step_threshold: int | None = None) -> None:
    filename_suffix = "_final" if final_step_threshold is not None else ""
    loss_path = save_dir / f"loss_vs_step_{wave_data.wave}{filename_suffix}.pdf"
    log_loss_path = save_dir / f"loss_vs_step_{wave_data.wave}_log{filename_suffix}.pdf"

    for save_path, log_scale in ((loss_path, False), (log_loss_path, True)):
        plt.figure(figsize=(10, 6))
        loss = select_curve(wave_data.loss, final_step_threshold)
        eval_loss = select_curve(wave_data.eval_loss, final_step_threshold)
        if loss is not None:
            draw_curve(loss, "Train Loss", WAVE_COLORS[wave_data.wave], "-")
        if eval_loss is not None:
            draw_curve(eval_loss, "Eval Loss", WAVE_COLORS[wave_data.wave], "--")
        suffix = " (Log Scale)" if log_scale else ""
        title_segment = f" from Step {final_step_threshold}" if final_step_threshold is not None else ""
        finish_plot(
            save_path,
            f"{wave_data.wave.capitalize()} K-Fold Average Train and Eval Loss{title_segment}{suffix}",
            "Loss (log scale)" if log_scale else "Loss",
            log_scale=log_scale,
        )

    if wave_data.eval_loss is not None:
        draw_metric_comparison(
            [wave_data],
            "eval_loss",
            "Eval Loss",
            save_dir / f"eval_loss_vs_step_{wave_data.wave}{filename_suffix}.pdf",
            final_step_threshold=final_step_threshold,
        )
    if wave_data.best_eval_loss is not None:
        draw_metric_comparison(
            [wave_data],
            "best_eval_loss",
            "Best Eval Loss",
            save_dir / f"best_eval_loss_vs_step_{wave_data.wave}{filename_suffix}.pdf",
            final_step_threshold=final_step_threshold,
        )


def draw_pairwise_against_sinusoid(
    waves_by_name: dict[str, WaveData],
    save_dir: Path,
    final_step_threshold: int | None = None,
) -> None:
    baseline = waves_by_name.get("sinusoid")
    if baseline is None:
        print("Skipping pairwise comparisons: sinusoid data is not loaded.")
        return

    filename_suffix = "_final" if final_step_threshold is not None else ""
    for wave, wave_data in waves_by_name.items():
        if wave == "sinusoid":
            continue

        for save_path, log_scale in (
            (save_dir / f"loss_vs_step_sinusoid_vs_{wave}{filename_suffix}.pdf", False),
            (save_dir / f"loss_vs_step_sinusoid_vs_{wave}_log{filename_suffix}.pdf", True),
        ):
            plt.figure(figsize=(10, 6))
            for current in (baseline, wave_data):
                color = WAVE_COLORS[current.wave]
                loss = select_curve(current.loss, final_step_threshold)
                eval_loss = select_curve(current.eval_loss, final_step_threshold)
                if loss is not None:
                    draw_curve(loss, f"{current.wave.capitalize()} Train Loss", color, "-")
                if eval_loss is not None:
                    draw_curve(eval_loss, f"{current.wave.capitalize()} Eval Loss", color, "--")
            suffix = " (Log Scale)" if log_scale else ""
            title_segment = f" from Step {final_step_threshold}" if final_step_threshold is not None else ""
            finish_plot(
                save_path,
                f"Sinusoid vs {wave.capitalize()} K-Fold Average Loss{title_segment}{suffix}",
                "Loss (log scale)" if log_scale else "Loss",
                log_scale=log_scale,
            )

        draw_metric_comparison(
            [baseline, wave_data],
            "eval_loss",
            "Eval Loss",
            save_dir / f"eval_loss_vs_step_sinusoid_vs_{wave}{filename_suffix}.pdf",
            final_step_threshold=final_step_threshold,
        )
        draw_metric_comparison(
            [baseline, wave_data],
            "best_eval_loss",
            "Best Eval Loss",
            save_dir / f"best_eval_loss_vs_step_sinusoid_vs_{wave}{filename_suffix}.pdf",
            final_step_threshold=final_step_threshold,
        )


def write_summary_csv(waves: Iterable[WaveData], save_path: Path) -> None:
    fieldnames = [
        "wave",
        "run_group",
        "folds",
        "train_points",
        "eval_points",
        "final_common_step",
        "final_train_loss_mean",
        "final_train_loss_std",
        "final_eval_loss_mean",
        "final_eval_loss_std",
        "final_best_eval_loss_mean",
        "final_best_eval_loss_std",
    ]
    with save_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for wave_data in waves:
            eval_curve = wave_data.eval_loss
            best_curve = wave_data.best_eval_loss
            writer.writerow(
                {
                    "wave": wave_data.wave,
                    "run_group": wave_data.group_dir.name,
                    "folds": len(wave_data.fold_tables),
                    "train_points": len(wave_data.loss.steps),
                    "eval_points": len(eval_curve.steps) if eval_curve is not None else 0,
                    "final_common_step": int(wave_data.loss.steps[-1]),
                    "final_train_loss_mean": wave_data.loss.mean[-1],
                    "final_train_loss_std": wave_data.loss.std[-1],
                    "final_eval_loss_mean": eval_curve.mean[-1] if eval_curve is not None else "",
                    "final_eval_loss_std": eval_curve.std[-1] if eval_curve is not None else "",
                    "final_best_eval_loss_mean": best_curve.mean[-1] if best_curve is not None else "",
                    "final_best_eval_loss_std": best_curve.std[-1] if best_curve is not None else "",
                }
            )
    print(f"Summary saved: {save_path}")


def main() -> None:
    args = parse_args()
    waves = parse_waves(args.waves)
    run_graph_dir = create_run_graph_dir(args.graphs_dir)

    waves_by_name = {}
    for wave in waves:
        wave_data = load_wave_data(args.outputs_dir, wave, args.run_id)
        waves_by_name[wave] = wave_data
        print(
            f"Using {wave}: {wave_data.group_dir} "
            f"({len(wave_data.fold_tables)} folds, {len(wave_data.loss.steps)} common train steps)"
        )

    loaded_waves = [waves_by_name[wave] for wave in waves]
    draw_loss_comparison(loaded_waves, run_graph_dir / "loss_vs_step_comparison.pdf")
    draw_loss_comparison(loaded_waves, run_graph_dir / "loss_vs_step_comparison_log.pdf", log_scale=True)
    draw_metric_comparison(loaded_waves, "eval_loss", "Eval Loss", run_graph_dir / "eval_loss_vs_step_comparison.pdf")
    draw_metric_comparison(
        loaded_waves,
        "best_eval_loss",
        "Best Eval Loss",
        run_graph_dir / "best_eval_loss_vs_step_comparison.pdf",
    )
    if not args.no_final_segment:
        final_step_threshold = args.final_step_threshold
        draw_loss_comparison(
            loaded_waves,
            run_graph_dir / "loss_vs_step_comparison_final.pdf",
            final_step_threshold=final_step_threshold,
        )
        draw_loss_comparison(
            loaded_waves,
            run_graph_dir / "loss_vs_step_comparison_log_final.pdf",
            log_scale=True,
            final_step_threshold=final_step_threshold,
        )
        draw_metric_comparison(
            loaded_waves,
            "eval_loss",
            "Eval Loss",
            run_graph_dir / "eval_loss_vs_step_comparison_final.pdf",
            final_step_threshold=final_step_threshold,
        )
        draw_metric_comparison(
            loaded_waves,
            "best_eval_loss",
            "Best Eval Loss",
            run_graph_dir / "best_eval_loss_vs_step_comparison_final.pdf",
            final_step_threshold=final_step_threshold,
        )

    if not args.no_individual:
        for wave_data in loaded_waves:
            draw_individual_wave(wave_data, run_graph_dir)
            if not args.no_final_segment:
                draw_individual_wave(wave_data, run_graph_dir, final_step_threshold=args.final_step_threshold)

    if not args.no_compare_sinusoid:
        draw_pairwise_against_sinusoid(waves_by_name, run_graph_dir)
        if not args.no_final_segment:
            draw_pairwise_against_sinusoid(
                waves_by_name,
                run_graph_dir,
                final_step_threshold=args.final_step_threshold,
            )

    write_summary_csv(loaded_waves, run_graph_dir / "averaged_metrics_summary.csv")


if __name__ == "__main__":
    main()
