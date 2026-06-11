#!/usr/bin/env python3
"""Compatibility wrapper for the root cramming k-fold launcher."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "train_k_fold_cross_validation.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
