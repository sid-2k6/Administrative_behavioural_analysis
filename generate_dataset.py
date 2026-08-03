#!/usr/bin/env python3
"""
generate_dataset.py
===================

Command-line entry point for generating the research-grade synthetic
multimodal educational administrative behaviour dataset.

Examples
--------
Generate the default 300-administrator x 12-month dataset::

    python generate_dataset.py

Generate a smaller, faster dataset into a custom directory::

    python generate_dataset.py --administrators 50 --months 6 \\
        --teachers 15 --output-dir output_small

Reproduce with a different seed and skip plots::

    python generate_dataset.py --seed 7 --no-plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

from edu_admin_synth.config import Config
from edu_admin_synth.pipeline import run_pipeline


def build_config_from_args(args: argparse.Namespace) -> Config:
    """Translate parsed CLI arguments into a :class:`Config`."""
    return Config(
        random_seed=args.seed,
        n_administrators=args.administrators,
        n_months=args.months,
        teachers_per_admin=args.teachers,
        output_dir=Path(args.output_dir),
        make_plots=not args.no_plots,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic multimodal educational administrative "
            "behaviour dataset (text + logs + surveys) for transformer-based "
            "fusion research."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--administrators", type=int, default=300,
                        help="Number of administrators to simulate.")
    parser.add_argument("--months", type=int, default=12,
                        help="Number of months per administrator.")
    parser.add_argument("--teachers", type=int, default=20,
                        help="Teacher survey respondents per administrator/month.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master random seed for full reproducibility.")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to write all generated artefacts to.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Disable generation of diagnostic plots.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress logging.")
    return parser.parse_args()


def main() -> None:
    """Program entry point."""
    args = parse_args()
    config = build_config_from_args(args)
    run_pipeline(config, verbose=not args.quiet)


if __name__ == "__main__":
    main()
