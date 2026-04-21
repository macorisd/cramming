import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


# Embed TrueType fonts in PDFs for better portability.
plt.rcParams["pdf.fonttype"] = "truetype"


REPO_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
GRAPHS_DIR = REPO_ROOT / "graphs"
WAVES = ("sinusoid", "triangular", "square", "sawtooth")
WAVE_COLORS = {
    "sinusoid": "r",
    "triangular": "g",
    "square": "b",
    "sawtooth": "orange",
}
VAL_LOSS_KEYS = ("val_loss", "valid_loss", "validation_loss", "test_loss", "eval_loss")


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parse_run_timestamp(csv_path: Path) -> datetime:
    run_dir = csv_path.parent
    timestamp = f"{run_dir.parent.name} {run_dir.name}"
    return datetime.strptime(timestamp, "%Y-%m-%d %H-%M-%S")


def find_latest_csv_for_wave(wave: str) -> Path:
    pattern = f"cramming_{wave}_single/pretrain/*/*/table_cramming_{wave}_single_convergence_results.csv"
    candidates = list(OUTPUTS_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No convergence CSV found for wave '{wave}'.")
    return max(candidates, key=parse_run_timestamp)


def extract_series(rows: List[Dict[str, str]], key: str) -> List[float]:
    values = []
    for row in rows:
        raw = row.get(key, "")
        if raw in ("", None):
            continue
        values.append(float(raw))
    return values


def find_validation_key(rows: List[Dict[str, str]]) -> Optional[str]:
    if not rows:
        return None
    keys = rows[0].keys()
    for key in VAL_LOSS_KEYS:
        if key in keys:
            return key
    return None


def draw_loss_comparison(
    series_by_wave: Dict[str, Dict[str, object]], save_path: Path, log_scale: bool = False
) -> None:
    plt.figure(figsize=(10, 6))

    validation_key = None
    for wave, info in series_by_wave.items():
        steps = info["steps"]
        train_loss = info["train_loss"]
        validation_key = validation_key or info["validation_key"]
        plt.plot(
            steps,
            train_loss,
            color=WAVE_COLORS[wave],
            linestyle="-",
            linewidth=2,
            label=f"{wave.capitalize()} Train Loss",
        )

        val_loss = info["val_loss"]
        if val_loss:
            plt.plot(
                steps[: len(val_loss)],
                val_loss,
                color=WAVE_COLORS[wave],
                linestyle="--",
                linewidth=2,
                label=f"{wave.capitalize()} Validation Loss",
            )

    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Loss (log scale)" if log_scale else "Loss", fontsize=12)
    title = "Training and Validation Loss vs Step" if validation_key else "Training Loss vs Step"
    if log_scale:
        title += " (Log Scale)"
    plt.title(title, fontsize=14)
    if log_scale:
        plt.yscale("log")
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, which="both", axis="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def draw_individual_wave_graph(
    wave: str, info: Dict[str, object], save_path: Path, log_scale: bool = False
) -> None:
    plt.figure(figsize=(10, 6))

    steps = info["steps"]
    train_loss = info["train_loss"]
    validation_key = info["validation_key"]
    plt.plot(
        steps,
        train_loss,
        color=WAVE_COLORS[wave],
        linestyle="-",
        linewidth=2,
        label=f"{wave.capitalize()} Train Loss",
    )

    val_loss = info["val_loss"]
    if val_loss:
        plt.plot(
            steps[: len(val_loss)],
            val_loss,
            color=WAVE_COLORS[wave],
            linestyle="--",
            linewidth=2,
            label=f"{wave.capitalize()} Validation Loss",
        )

    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Loss (log scale)" if log_scale else "Loss", fontsize=12)
    title = (
        f"{wave.capitalize()} Training and Validation Loss vs Step"
        if validation_key
        else f"{wave.capitalize()} Training Loss vs Step"
    )
    if log_scale:
        title += " (Log Scale)"
    plt.title(title, fontsize=14)
    if log_scale:
        plt.yscale("log")
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, which="both", axis="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def draw_sinusoid_comparison(
    baseline_wave: str,
    baseline_info: Dict[str, object],
    compare_wave: str,
    compare_info: Dict[str, object],
    save_path: Path,
    final_only: bool = False,
    final_step_threshold: int = 600000,
) -> None:
    plt.figure(figsize=(10, 6))

    for wave, info in ((baseline_wave, baseline_info), (compare_wave, compare_info)):
        steps = info["steps"]
        train_loss = info["train_loss"]

        if final_only:
            filtered_pairs = [(step, loss) for step, loss in zip(steps, train_loss) if step >= final_step_threshold]
            if not filtered_pairs:
                continue
            steps, train_loss = zip(*filtered_pairs)

        plt.plot(
            steps,
            train_loss,
            color=WAVE_COLORS[wave],
            linestyle="-",
            linewidth=2,
            label=f"{wave.capitalize()} Train Loss",
        )

    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Loss (log scale)", fontsize=12)
    title_suffix = f" from Step {final_step_threshold}" if final_only else ""
    plt.title(
        f"{baseline_wave.capitalize()} vs {compare_wave.capitalize()} Training Loss{title_suffix} (Log Scale)",
        fontsize=14,
    )
    plt.yscale("log")
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, which="both", axis="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot loss curves for positional wave experiments.")
    parser.add_argument(
        "-i",
        "--individual",
        action="store_true",
        help="Also save one individual loss graph per wave.",
    )
    parser.add_argument(
        "--compare-sinusoid",
        action="store_true",
        help="Also save logarithmic comparisons of each non-sinusoid wave against sinusoid.",
    )
    parser.add_argument(
        "-f",
        "--final-only",
        action="store_true",
        help="With --compare-sinusoid, also save the same comparisons restricted to the final training segment.",
    )
    args = parser.parse_args()
    if args.final_only and not args.compare_sinusoid:
        parser.error("-f/--final-only requires --compare-sinusoid.")
    return args


def create_run_graph_dir() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_graph_dir = GRAPHS_DIR / timestamp
    run_graph_dir.mkdir(parents=True, exist_ok=False)
    return run_graph_dir


def main() -> None:
    args = parse_args()
    GRAPHS_DIR.mkdir(exist_ok=True)
    run_graph_dir = create_run_graph_dir()
    save_path = run_graph_dir / "loss_vs_step_comparison.pdf"
    log_save_path = run_graph_dir / "loss_vs_step_comparison_log.pdf"
    series_by_wave = {}

    for wave in WAVES:
        csv_path = find_latest_csv_for_wave(wave)
        rows = load_rows(csv_path)
        validation_key = find_validation_key(rows)
        series_by_wave[wave] = {
            "csv_path": csv_path,
            "steps": extract_series(rows, "step"),
            "train_loss": extract_series(rows, "loss"),
            "validation_key": validation_key,
            "val_loss": extract_series(rows, validation_key) if validation_key else [],
        }
        print(f"Using {wave}: {csv_path}")

    draw_loss_comparison(series_by_wave, save_path)
    print(f"Graph saved: {save_path}")

    draw_loss_comparison(series_by_wave, log_save_path, log_scale=True)
    print(f"Graph saved: {log_save_path}")

    if args.individual:
        for wave, info in series_by_wave.items():
            individual_save_path = run_graph_dir / f"loss_vs_step_{wave}.pdf"
            draw_individual_wave_graph(wave, info, individual_save_path)
            print(f"Graph saved: {individual_save_path}")

            individual_log_save_path = run_graph_dir / f"loss_vs_step_{wave}_log.pdf"
            draw_individual_wave_graph(wave, info, individual_log_save_path, log_scale=True)
            print(f"Graph saved: {individual_log_save_path}")

    if args.compare_sinusoid:
        sinusoid_info = series_by_wave["sinusoid"]
        for wave in WAVES:
            if wave == "sinusoid":
                continue

            comparison_save_path = run_graph_dir / f"loss_vs_step_sinusoid_vs_{wave}_log.pdf"
            draw_sinusoid_comparison(
                "sinusoid",
                sinusoid_info,
                wave,
                series_by_wave[wave],
                comparison_save_path,
            )
            print(f"Graph saved: {comparison_save_path}")

            if args.final_only:
                final_comparison_save_path = run_graph_dir / f"loss_vs_step_sinusoid_vs_{wave}_log_final.pdf"
                draw_sinusoid_comparison(
                    "sinusoid",
                    sinusoid_info,
                    wave,
                    series_by_wave[wave],
                    final_comparison_save_path,
                    final_only=True,
                )
                print(f"Graph saved: {final_comparison_save_path}")


if __name__ == "__main__":
    main()
