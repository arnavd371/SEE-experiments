"""
Main entry point for the SEE experiment suite.

Usage:
  python run_all.py                  # full experiment
  python run_all.py --fast           # quick smoke test (~5 min)
  python run_all.py --part 1         # only Part 1
  python run_all.py --part 2         # only Part 2 (requires best_lrs.yaml)
  python run_all.py --part 3         # only Part 3 (requires best_lrs.yaml)
  python run_all.py --part 4         # only Part 4 (requires parts 1 & 3)
  python run_all.py --fast --part 1  # fast Part 1
"""
import argparse
import os
import sys
import time
from pathlib import Path

import yaml
import pandas as pd

import config
from src.utils.seeding import set_all_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true',
                        help='Use FAST_ parameters for quick smoke test')
    parser.add_argument('--part', type=int, default=None, choices=[1, 2, 3, 4],
                        help='Run only this part (default: all)')
    args = parser.parse_args()

    fast = args.fast
    part = args.part

    print(f"Device: {config.DEVICE}")
    print(f"Fast mode: {fast}")
    if fast:
        print(f"  N_TRIALS={config.FAST_N_TRIALS}, T_MAX={config.FAST_T_MAX}, "
              f"DIMS={config.FAST_DIMENSIONS}, NN_STEPS={config.FAST_NN_STEPS}")

    Path('results').mkdir(exist_ok=True)
    set_all_seeds(config.GLOBAL_SEED)

    t_start = time.time()

    df1, best_lrs, df2, df3 = None, None, None, None

    # ------------------------------------------------------------------ Part 1
    if part is None or part == 1:
        print("\n" + "=" * 60)
        print("PART 1: Vectorized 2D benchmark experiments")
        print("=" * 60)
        from src.experiments.part1_vectorized import run_part1
        df1, best_lrs = run_part1(fast=fast)

    # Load best_lrs if Part 1 was skipped
    if best_lrs is None:
        lr_path = Path('results/best_lrs.yaml')
        if lr_path.exists():
            with open(lr_path) as f:
                best_lrs = yaml.safe_load(f)
            print(f"Loaded best_lrs from {lr_path}")
        else:
            print("WARNING: best_lrs.yaml not found; using lr=0.01 for all optimizers")
            best_lrs = {k: 0.01 for k in config.OPTIMIZERS}

    # ------------------------------------------------------------------ Part 2
    if part is None or part == 2:
        print("\n" + "=" * 60)
        print("PART 2: High-dimensional scaling")
        print("=" * 60)
        from src.experiments.part2_highdim import run_part2
        df2 = run_part2(best_lrs, fast=fast)

    # ------------------------------------------------------------------ Part 3
    if part is None or part == 3:
        print("\n" + "=" * 60)
        print("PART 3: Neural network experiments")
        print("=" * 60)
        from src.experiments.part3_nn import run_part3
        df3 = run_part3(best_lrs, fast=fast)

    # ------------------------------------------------------------------ Part 4
    if part is None or part == 4:
        print("\n" + "=" * 60)
        print("PART 4: Synthesis and ablations")
        print("=" * 60)

        # Load data if not computed in this run
        if df1 is None:
            p = Path('results/part1.csv')
            if p.exists():
                df1 = pd.read_csv(p)
            else:
                print("ERROR: results/part1.csv not found. Run Part 1 first.")
                sys.exit(1)

        if df3 is None:
            p = Path('results/part3.csv')
            if p.exists():
                df3 = pd.read_csv(p)
            else:
                print("WARNING: results/part3.csv not found; Part 4C will be empty")
                df3 = pd.DataFrame(columns=['task', 'optimizer', 'run_seed', 'step',
                                            'loss', 'grad_norm', 'is_saddle',
                                            'SEE_NN', 'SEE_NN_CI_lo', 'SEE_NN_CI_hi',
                                            'lambda_min', 'lambda_max',
                                            'steps_to_convergence',
                                            'final_loss', 'final_accuracy'])

        from src.experiments.part4_synthesis import run_part4
        df4a, df4b, df4c, df4d = run_part4(df1, df3, best_lrs, fast=fast)

        # ---------------------------------------------------------------- Figures
        print("\n" + "=" * 60)
        print("FIGURES")
        print("=" * 60)

        if df2 is None:
            p = Path('results/part2.csv')
            if p.exists():
                df2 = pd.read_csv(p)
            else:
                df2 = pd.DataFrame()

        from src.plots.all_figures import make_all_figures
        make_all_figures(df1, df2, df3, df4b, df4d)

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed/60:.1f} min")
    print("Done.")


if __name__ == '__main__':
    main()
