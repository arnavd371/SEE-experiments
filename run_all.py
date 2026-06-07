#!/usr/bin/env python3
"""
Master runner for all SEE experiments.

Usage:
  python run_all.py                  # Full run, all parts
  python run_all.py --fast           # Fast smoke-test (< 5 min)
  python run_all.py --part 1         # Run only Part 1
  python run_all.py --part 1 --resume  # Resume Part 1 from checkpoint
  python run_all.py --part 1 2 3     # Run parts 1, 2, and 3
  python run_all.py --plots-only     # Regenerate all figures from CSVs
"""

import argparse
import os
import sys
import time

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.utils.logging_utils import log_environment


def parse_args():
    parser = argparse.ArgumentParser(
        description='SEE: Saddle Escape Efficiency experiment suite')
    parser.add_argument('--fast', action='store_true',
                        help='Quick smoke-test: N=50, T_max=200, GPT_steps=500')
    parser.add_argument('--part', nargs='+', type=int, default=None,
                        help='Run specific parts (e.g. --part 1 2). Default: all.')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint (Parts 1 and 2).')
    parser.add_argument('--plots-only', action='store_true',
                        help='Regenerate all figures from existing CSVs.')
    return parser.parse_args()


def apply_fast_overrides():
    """Monkey-patch config with fast-mode values."""
    config.N_TRIALS_PART1 = config.FAST_N_TRIALS
    config.N_TRIALS_PART2 = config.FAST_N_TRIALS
    config.N_TRIALS_PART2_HIGHD = config.FAST_N_TRIALS
    config.T_MAX = config.FAST_T_MAX
    config.T_MAX_HIGHD = config.FAST_T_MAX
    config.DIMENSIONS = [d for d in config.DIMENSIONS if d <= config.FAST_D_MAX]
    config.LLM_STEPS = config.FAST_GPT_STEPS
    config.NN_STEPS = config.FAST_NN_STEPS
    config.NN_RUNS = config.FAST_NN_RUNS
    config.LLM_RUNS = config.FAST_LLM_RUNS
    config.BOOTSTRAP_RESAMPLES = config.FAST_BOOTSTRAP
    config.GRID_SIZE = config.FAST_GRID_SIZE
    config.LLM_WARMUP_STEPS = min(20, config.FAST_GPT_STEPS // 10)
    config.LLM_PLATEAU_CHECK_INTERVAL = max(10, config.FAST_GPT_STEPS // 20)
    config.NN_SUBTRIAL_N = 10          # very few sub-trials in fast mode
    config.NN_SUBTRIAL_T_MAX = 50
    print('[FAST MODE] Overrides applied:')
    print(f'  N_TRIALS={config.N_TRIALS_PART1}, T_MAX={config.T_MAX}')
    print(f'  DIMENSIONS={config.DIMENSIONS}')
    print(f'  LLM_STEPS={config.LLM_STEPS}, NN_STEPS={config.NN_STEPS}')


def run_part1(fast, resume):
    print('\n' + '═' * 60)
    print('PART 1: Classical 2D Benchmark Experiments')
    print('═' * 60)
    from src.experiments.part1_classical import run_part1 as _run
    t0 = time.time()
    df = _run(fast=fast, resume=resume)
    print(f'Part 1 completed in {time.time() - t0:.1f}s')
    return df


def run_part2(fast, resume):
    print('\n' + '═' * 60)
    print('PART 2: High-Dimensional Scaling')
    print('═' * 60)
    from src.experiments.part2_highdim import run_part2 as _run
    t0 = time.time()
    df = _run(fast=fast, resume=resume)
    print(f'Part 2 completed in {time.time() - t0:.1f}s')
    return df


def run_part3(fast):
    print('\n' + '═' * 60)
    print('PART 3: Neural Network — Real Datasets')
    print('═' * 60)
    from src.experiments.part3_nn import run_part3 as _run
    t0 = time.time()
    df = _run(fast=fast)
    print(f'Part 3 completed in {time.time() - t0:.1f}s')
    return df


def run_part4(fast):
    print('\n' + '═' * 60)
    print('PART 4: LLM Training Proxy (Tiny GPT on Wikitext-2)')
    print('═' * 60)
    from src.experiments.part4_llm_proxy import run_part4 as _run
    t0 = time.time()
    df = _run(fast=fast)
    print(f'Part 4 completed in {time.time() - t0:.1f}s')
    return df


def run_part5(fast):
    print('\n' + '═' * 60)
    print('PART 5: Predictive Validity')
    print('═' * 60)
    from src.experiments.part5_synthesis import run_part5 as _run
    t0 = time.time()
    df = _run(fast=fast)
    print(f'Part 5 completed in {time.time() - t0:.1f}s')
    return df


def generate_all_figures():
    print('\n' + '═' * 60)
    print('GENERATING ALL FIGURES')
    print('═' * 60)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    from src.plots.figure1_heatmaps import plot_figure1
    from src.plots.figure2_dimension import plot_figure2
    from src.plots.figure3_saddle_index import plot_figure3
    from src.plots.figure4_nn_training import plot_figure4
    from src.plots.figure5_main_result import plot_figure5
    from src.plots.figure6_escape_types import plot_figure6
    from src.plots.figure7_escape_time import plot_figure7

    plot_figure1()
    plot_figure2()
    plot_figure3()
    plot_figure4()
    plot_figure5()
    plot_figure6()
    plot_figure7()
    print(f'\nAll figures saved to {config.FIGURES_DIR}')


def main():
    args = parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    # Log environment
    env_log = os.path.join(config.RESULTS_DIR, 'env_log.txt')
    log_environment(env_log)
    print(f'\nDevice: {config.DEVICE}')

    if args.fast:
        apply_fast_overrides()

    if args.plots_only:
        generate_all_figures()
        return

    parts_to_run = set(args.part) if args.part else {1, 2, 3, 4, 5}

    t_total = time.time()

    if 1 in parts_to_run:
        run_part1(args.fast, args.resume)

    if 2 in parts_to_run:
        run_part2(args.fast, args.resume)

    if 3 in parts_to_run:
        if not os.path.exists(config.BEST_LRS_PATH):
            print('WARNING: best_lrs.yaml not found. Run Part 1 first, or '
                  'using default LR=0.01 for all optimizers.')
        run_part3(args.fast)

    if 4 in parts_to_run:
        if not os.path.exists(config.BEST_LRS_PATH):
            print('WARNING: best_lrs.yaml not found. Run Part 1 first.')
        run_part4(args.fast)

    if 5 in parts_to_run:
        run_part5(args.fast)

    # Generate figures after experiments
    if parts_to_run:
        generate_all_figures()

    elapsed = time.time() - t_total
    print(f'\n{"═" * 60}')
    print(f'All requested parts completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)')
    print(f'Results:  {config.RESULTS_DIR}')
    print(f'Figures:  {config.FIGURES_DIR}')
    print('═' * 60)


if __name__ == '__main__':
    main()
