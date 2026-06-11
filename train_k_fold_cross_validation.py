#!/usr/bin/env python3
"""Launch reproducible k-fold cramming pretraining across positional waves.

This mirrors the CLI style of the basic transformer k-fold launcher while
reusing ``pretrain.py`` and Hydra overrides. Each fold keeps the same global
training seed and the same deterministic fold split seed so every wave is
trained under identical stochastic conditions.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


SUPPORTED_WAVES = ("sinusoid", "triangular", "square", "sawtooth")


@dataclass(frozen=True)
class PlannedRun:
    wave: str
    fold: int

    @property
    def run_name(self) -> str:
        return f"cramming_{self.wave}_kfold"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Skip folds that already finished successfully.")
    parser.add_argument(
        "--functions",
        type=str,
        default=None,
        help="Comma-separated periodic functions. Defaults to all supported waves.",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Comma-separated fold numbers to train (1-based, e.g. 1,2,3). Defaults to all folds.",
    )
    parser.add_argument("--k-folds", type=int, default=10, help="Total number of folds used to split the dataset.")
    parser.add_argument("--seed", type=int, default=89, help="Global training seed shared by every wave and fold.")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed used to build the deterministic fold split.")
    parser.add_argument(
        "--run-id",
        default=os.environ.get("KFOLD_RUN_ID"),
        help="Shared timestamp-like identifier used to group all folds of the same k-fold wave run.",
    )
    parser.add_argument("--arch", default="crammed-bert", help="Hydra architecture config passed to pretrain.py.")
    parser.add_argument("--train", default="bert-base", help="Hydra training config passed to pretrain.py.")
    parser.add_argument("--data", default="bookcorpus-wikipedia", help="Hydra data config passed to pretrain.py.")
    parser.add_argument("--budget", type=float, default=24.0, help="Central training budget in hours.")
    parser.add_argument("--base-dir", default="outputs-kfold", help="Base output directory passed to Hydra.")
    parser.add_argument(
        "--impl-path",
        default=os.environ.get("CRAMMING_DATASET_CACHE_DIR", os.environ.get("DATASET_CACHE_DIR", "outputs/data")),
        help="Path where cramming stores or loads processed datasets.",
    )
    parser.add_argument("--impl-threads", type=int, default=16, help="Number of preprocessing / loading threads.")
    parser.add_argument("--validation-interval", type=int, default=1000, help="Run eval every N train steps.")
    parser.add_argument(
        "--validation-max-batches",
        type=int,
        default=None,
        help="Optional cap on validation batches for each eval pass.",
    )
    parser.add_argument("--print-loss-every", type=int, default=1000, help="Print averaged loss every N steps.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch pretrain.py.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override to append verbatim to pretrain.py.",
    )
    parser.add_argument("--dryrun", action="store_true", help="Pass dryrun=True to pretrain.py.")
    parser.add_argument(
        "--launcher-dry-run",
        action="store_true",
        help="Print the generated pretrain.py commands without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep launching remaining folds even if one wave/fold fails.",
    )
    args = parser.parse_args()

    if args.k_folds < 2:
        parser.error("--k-folds must be >= 2.")
    return args


def parse_waves(raw_waves: str | None) -> list[str]:
    if raw_waves is None:
        return list(SUPPORTED_WAVES)

    waves = [wave.strip() for wave in raw_waves.split(",") if wave.strip()]
    if not waves:
        raise ValueError("At least one periodic function must be provided.")

    invalid = sorted(set(waves) - set(SUPPORTED_WAVES))
    if invalid:
        raise ValueError(f"Unsupported functions: {', '.join(invalid)}. Supported: {', '.join(SUPPORTED_WAVES)}.")
    return waves


def parse_folds(raw_folds: str | None, k_folds: int) -> list[int]:
    if raw_folds is None:
        return list(range(1, k_folds + 1))

    try:
        folds = [int(fold.strip()) for fold in raw_folds.split(",") if fold.strip()]
    except ValueError as exc:
        raise ValueError("--folds must be a comma-separated list of integers.") from exc

    if not folds:
        raise ValueError("At least one fold must be provided.")
    invalid = [fold for fold in folds if fold < 1 or fold > k_folds]
    if invalid:
        raise ValueError(f"Fold numbers must be between 1 and {k_folds}. Invalid values: {invalid}.")
    return folds


def resolve_base_dir(repo_root: Path, base_dir: str) -> Path:
    target = Path(base_dir)
    if not target.is_absolute():
        target = repo_root / target
    return target


def resolve_dataset_display_name(repo_root: Path, data_config_name: str) -> str:
    data_config_path = repo_root / "cramming" / "config" / "data" / f"{data_config_name}.yaml"
    if not data_config_path.is_file():
        return data_config_name

    with data_config_path.open() as handle:
        data_cfg = yaml.safe_load(handle) or {}
    return str(data_cfg.get("name", data_config_name))


def normalize_run_id(run_id: str | None) -> str:
    if run_id:
        return run_id
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def wave_root_dir(base_dir: Path, run_name: str) -> Path:
    return base_dir / run_name


def group_dir_name(run_id: str, dataset_name: str) -> str:
    return f"{run_id}_{dataset_name}"


def fold_dir(base_dir: Path, run_name: str, run_group_dir_name: str, fold: int) -> Path:
    return wave_root_dir(base_dir, run_name) / run_group_dir_name / f"fold_{fold}"


def list_group_run_dirs(wave_root: Path) -> list[Path]:
    if not wave_root.is_dir():
        return []
    return sorted(path for path in wave_root.iterdir() if path.is_dir() and path.name != "checkpoints")


def fold_dir_is_complete(fold_dir_path: Path, run_name: str) -> bool:
    convergence_table = fold_dir_path / f"table_{run_name}_convergence_results.csv"
    checkpoint_root = fold_dir_path / "checkpoints"
    has_checkpoint = checkpoint_root.is_dir() and any(
        checkpoint_dir.is_dir() and (checkpoint_dir / "model.safetensors").is_file()
        for checkpoint_dir in checkpoint_root.iterdir()
    )
    return convergence_table.is_file() and has_checkpoint


def latest_group_dir(base_dir: Path, run_name: str) -> Path | None:
    group_dirs = list_group_run_dirs(wave_root_dir(base_dir, run_name))
    return group_dirs[-1] if group_dirs else None


def resolve_group_dir(
    base_dir: Path,
    run_name: str,
    dataset_name: str,
    args: argparse.Namespace,
    generated_run_id: str,
) -> Path:
    if args.run_id:
        return wave_root_dir(base_dir, run_name) / group_dir_name(args.run_id, dataset_name)
    if args.resume:
        existing = latest_group_dir(base_dir, run_name)
        if existing is not None:
            return existing
    return wave_root_dir(base_dir, run_name) / group_dir_name(generated_run_id, dataset_name)


def cleanup_incomplete_fold_dir(fold_dir_path: Path, run_name: str) -> None:
    if fold_dir_path.is_dir() and not fold_dir_is_complete(fold_dir_path, run_name):
        shutil.rmtree(fold_dir_path)


def build_command(args: argparse.Namespace, base_dir: Path, run: PlannedRun, run_group_dir: Path) -> list[str]:
    target_fold_dir = run_group_dir / f"fold_{run.fold}"
    command = [
        args.python,
        "pretrain.py",
        f"name={run.run_name}",
        f"base_dir={base_dir}",
        f"hydra.run.dir={target_fold_dir}",
        f"hydra.sweep.dir={target_fold_dir}",
        f"arch={args.arch}",
        f"train={args.train}",
        f"data={args.data}",
        f"budget={args.budget}",
        f"seed={args.seed}",
        f"impl.path={args.impl_path}",
        f"impl.threads={args.impl_threads}",
        "impl.compile_torch=False",
        f"impl.print_loss_every_nth_step={args.print_loss_every}",
        "impl.enable_huggingface_offline_mode=True",
        "validation.enabled=True",
        f"validation.interval={args.validation_interval}",
        "validation.fixed_masks=True",
        f"arch.embedding.positional_wave={run.wave}",
        "k_fold.enabled=True",
        f"k_fold.num_folds={args.k_folds}",
        f"k_fold.current_fold={run.fold}",
        f"k_fold.seed={args.split_seed}",
    ]
    if args.validation_max_batches is not None:
        command.append(f"validation.max_batches={args.validation_max_batches}")
    if args.dryrun:
        command.append("dryrun=True")
    command.extend(args.override)
    return command


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    base_dir = resolve_base_dir(repo_root, args.base_dir)
    dataset_name = resolve_dataset_display_name(repo_root, args.data)
    waves = parse_waves(args.functions)
    folds = parse_folds(args.folds, args.k_folds)
    planned_runs = [PlannedRun(wave=wave, fold=fold) for wave in waves for fold in folds]
    resolved_run_id = normalize_run_id(args.run_id)

    print("#" * 88)
    print("# CRAMMING K-FOLD CROSS VALIDATION")
    print(f"# Repo root: {repo_root}")
    print(f"# Base dir: {base_dir}")
    print(f"# Dataset display name: {dataset_name}")
    print(f"# Run id: {resolved_run_id}")
    print(f"# Functions: {', '.join(waves)}")
    print(f"# Folds: {', '.join(str(fold) for fold in folds)}")
    print(f"# Total folds requested: {len(planned_runs)}")
    print(f"# Global training seed: {args.seed}")
    print(f"# Fold split seed: {args.split_seed}")
    if args.resume:
        print("# Resume mode: enabled")
    if args.launcher_dry_run:
        print("# Launcher dry-run mode: enabled")
    print("#" * 88)

    failures: list[tuple[PlannedRun, int]] = []
    skipped = 0

    for idx, run in enumerate(planned_runs, start=1):
        run_group_dir = resolve_group_dir(base_dir, run.run_name, dataset_name, args, resolved_run_id)
        target_fold_dir = run_group_dir / f"fold_{run.fold}"
        command = build_command(args, base_dir, run, run_group_dir)

        if args.resume:
            if fold_dir_is_complete(target_fold_dir, run.run_name):
                skipped += 1
                print(f"[{idx}/{len(planned_runs)}] Skipping completed run {run.run_name} fold {run.fold}")
                continue
            cleanup_incomplete_fold_dir(target_fold_dir, run.run_name)

        print(f"[{idx}/{len(planned_runs)}] {run.wave} | fold {run.fold} | {run_group_dir.name}/fold_{run.fold}")
        print(f"  {shlex.join(command)}")
        if args.launcher_dry_run:
            continue

        completed = subprocess.run(command, cwd=repo_root, check=False)
        if completed.returncode != 0:
            failures.append((run, completed.returncode))
            print(f"  Run failed with exit code {completed.returncode}.")
            if not args.continue_on_error:
                break

    print("#" * 88)
    print(f"# Requested runs: {len(planned_runs)}")
    print(f"# Skipped completed runs: {skipped}")
    print(f"# Failed runs: {len(failures)}")
    print("#" * 88)

    if failures:
        for run, returncode in failures:
            print(f"- wave={run.wave} fold={run.fold} exit_code={returncode}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
