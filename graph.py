import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


# Embed TrueType fonts in PDFs for better portability.
plt.rcParams["pdf.fonttype"] = "truetype"


REPO_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
SAVE_PATH = REPO_ROOT / "loss_vs_step_comparison.pdf"
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


def draw_loss_comparison(series_by_wave: Dict[str, Dict[str, object]], save_path: Path) -> None:
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
    plt.ylabel("Loss", fontsize=12)
    title = "Training and Validation Loss vs Step" if validation_key else "Training Loss vs Step"
    plt.title(title, fontsize=14)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, which="both", axis="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
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

    draw_loss_comparison(series_by_wave, SAVE_PATH)
    print(f"Graph saved: {SAVE_PATH}")


if __name__ == "__main__":
    main()
