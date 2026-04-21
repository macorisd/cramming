#!/usr/bin/env python3
"""Launch k-fold cramming pretraining runs across positional waves.

This wrapper intentionally reuses ``pretrain.py`` and Hydra overrides instead of
duplicating the training loop. For a real cross-validation setup, point
``--fold-data-template`` to fold-specific Hydra data configs such as
``my-corpus-fold{fold}``. If you provide ``--data`` instead, the script will
repeat training across folds with isolated names and seeds, but without changing
the underlying data split.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

SUPPORTED_WAVES = ("sinusoid", "triangular", "square", "sawtooth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-name", required=True, help="Prefix used to build Hydra run names.")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds to execute.")
    parser.add_argument("--start-fold", type=int, default=1, help="First fold index to execute (1-based).")
    parser.add_argument(
        "--waves",
        default=",".join(SUPPORTED_WAVES),
        help="Comma-separated positional waves to run. Defaults to all supported waves.",
    )
    parser.add_argument("--arch", default="crammed-bert", help="Hydra architecture config passed to pretrain.py.")
    parser.add_argument("--train", default="bert-o4", help="Hydra training config passed to pretrain.py.")
    parser.add_argument("--impl", help="Optional Hydra implementation config.")
    parser.add_argument("--data", help="Hydra data config shared by every fold.")
    parser.add_argument(
        "--fold-data-template",
        help="Fold-specific Hydra data config template, for example my-corpus-fold{fold}.",
    )
    parser.add_argument("--base-dir", default="outputs", help="Base output directory passed to Hydra.")
    parser.add_argument(
        "--seed-base",
        type=int,
        default=1234,
        help="Base seed used to derive per-fold seeds as seed_base + fold - 1.",
    )
    parser.add_argument("--budget", type=float, help="Optional Hydra budget override in hours.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch pretrain.py.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override. Supports {fold}, {wave}, and {run_name} placeholders.",
    )
    parser.add_argument(
        "--fold-override",
        action="append",
        default=[],
        help="Extra fold-specific Hydra override template. Supports {fold}, {wave}, and {run_name}.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose output folder already exists under base_dir/run_name.",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Pass dryrun=True to Hydra so cramming performs a smoke-test training run.",
    )
    parser.add_argument(
        "--launcher-dry-run",
        action="store_true",
        help="Print the generated commands without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep launching remaining runs even if one fold/wave fails.",
    )
    args = parser.parse_args()

    if args.folds < 1:
        parser.error("--folds must be >= 1.")
    if args.start_fold < 1:
        parser.error("--start-fold must be >= 1.")
    if args.data and args.fold_data_template:
        parser.error("Choose either --data or --fold-data-template, not both.")
    if not args.data and not args.fold_data_template:
        parser.error("Provide either --data or --fold-data-template.")
    return args


def parse_waves(raw_waves: str) -> list[str]:
    waves = [wave.strip() for wave in raw_waves.split(",") if wave.strip()]
    if not waves:
        raise ValueError("At least one positional wave must be provided.")

    invalid = sorted(set(waves) - set(SUPPORTED_WAVES))
    if invalid:
        raise ValueError(f"Unsupported waves: {', '.join(invalid)}. Supported: {', '.join(SUPPORTED_WAVES)}.")
    return waves


def format_template(template: str, *, fold: int, wave: str, run_name: str) -> str:
    return template.format(fold=fold, wave=wave, run_name=run_name)


def resolve_run_dir(repo_root: Path, base_dir: str, run_name: str) -> Path:
    target = Path(base_dir)
    if not target.is_absolute():
        target = repo_root / target
    return target / run_name


def build_command(args: argparse.Namespace, *, fold: int, wave: str, run_name: str) -> list[str]:
    command = [
        args.python,
        "pretrain.py",
        f"name={run_name}",
        f"base_dir={args.base_dir}",
        f"arch={args.arch}",
        f"train={args.train}",
        f"arch.embedding.positional_wave={wave}",
        f"seed={args.seed_base + fold - 1}",
    ]
    if args.impl:
        command.append(f"impl={args.impl}")
    if args.budget is not None:
        command.append(f"budget={args.budget}")
    if args.dryrun:
        command.append("dryrun=True")

    if args.fold_data_template:
        command.append(f"data={format_template(args.fold_data_template, fold=fold, wave=wave, run_name=run_name)}")
    elif args.data:
        command.append(f"data={args.data}")

    for override in args.override:
        command.append(format_template(override, fold=fold, wave=wave, run_name=run_name))
    for override in args.fold_override:
        command.append(format_template(override, fold=fold, wave=wave, run_name=run_name))
    return command


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    waves = parse_waves(args.waves)
    folds = range(args.start_fold, args.start_fold + args.folds)

    planned_runs = [(wave, fold) for wave in waves for fold in folds]
    failures = []

    print(f"# Repo root: {repo_root}")
    print(f"# Waves: {', '.join(waves)}")
    print(f"# Folds: {', '.join(str(fold) for fold in folds)}")
    print(f"# Total runs: {len(planned_runs)}")

    for idx, (wave, fold) in enumerate(planned_runs, start=1):
        run_name = f"{args.base_name}_{wave}_fold{fold:02d}"
        run_dir = resolve_run_dir(repo_root, args.base_dir, run_name)
        command = build_command(args, fold=fold, wave=wave, run_name=run_name)

        if args.skip_existing and run_dir.exists():
            print(f"[{idx}/{len(planned_runs)}] Skipping existing run directory: {run_dir}")
            continue

        print(f"[{idx}/{len(planned_runs)}] {wave} | fold {fold} | {run_name}")
        print(f"  {shlex.join(command)}")
        if args.launcher_dry_run:
            continue

        completed = subprocess.run(command, cwd=repo_root, check=False)
        if completed.returncode != 0:
            failures.append((wave, fold, completed.returncode))
            print(f"  Run failed with exit code {completed.returncode}.")
            if not args.continue_on_error:
                break

    if failures:
        print("# Failed runs:")
        for wave, fold, returncode in failures:
            print(f"- wave={wave} fold={fold} exit_code={returncode}")
        return 1

    print("# All requested runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
