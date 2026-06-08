"""
run_all.py — master entry point.

Usage:
  python run_all.py                  # full run
  python run_all.py --fast           # quick smoke-test
  python run_all.py --part 1         # only Part 1
  python run_all.py --fast --part 2  # fast Part 2
"""
import argparse
import sys
import yaml
from pathlib import Path

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--fast",   action="store_true")
parser.add_argument("--part",   type=int, default=0,
                    help="0 = all parts; 1-4 = individual part")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
from config import Config, apply_fast
if args.fast:
    apply_fast(Config)
    print("=== FAST mode ===")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
FIGS_DIR    = ROOT / "results" / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Seeding ───────────────────────────────────────────────────────────────────
from src.utils.seeding import set_all_seeds
set_all_seeds(42)

# ── Part helpers ──────────────────────────────────────────────────────────────
def _load_best_lrs():
    p = RESULTS_DIR / "best_lrs.yaml"
    if p.exists():
        with open(p) as fh:
            return yaml.safe_load(fh)
    return None


def run_part1():
    print("\n" + "=" * 60)
    print("PART 1: Vectorized 2-D benchmarks")
    print("=" * 60)
    from src.experiments.part1_vectorized import run_part1 as _run
    df, escape_data = _run(Config, RESULTS_DIR)
    return df


def run_part2():
    print("\n" + "=" * 60)
    print("PART 2: High-dimensional scaling")
    print("=" * 60)
    from src.experiments.part2_highdim import run_part2 as _run
    return _run(Config, RESULTS_DIR, best_lrs=_load_best_lrs())


def run_part3():
    print("\n" + "=" * 60)
    print("PART 3: Neural-network experiments")
    print("=" * 60)
    from src.experiments.part3_nn import run_part3 as _run
    return _run(Config, RESULTS_DIR, best_lrs=_load_best_lrs())


def run_part4(df1=None, df3=None):
    print("\n" + "=" * 60)
    print("PART 4: Synthesis & ablations")
    print("=" * 60)
    from src.experiments.part4_synthesis import run_part4 as _run
    return _run(Config, RESULTS_DIR, df1=df1, df3=df3)


def run_figures():
    print("\n" + "=" * 60)
    print("FIGURES")
    print("=" * 60)
    best_lrs = _load_best_lrs() or {}
    from src.plots.all_figures import generate_all
    generate_all(RESULTS_DIR, FIGS_DIR, best_lrs)


# ── Main ──────────────────────────────────────────────────────────────────────
df1 = df3 = None

if args.part in (0, 1):
    df1 = run_part1()

if args.part in (0, 2):
    run_part2()

if args.part in (0, 3):
    df3 = run_part3()

if args.part in (0, 4):
    run_part4(df1=df1, df3=df3)

if args.part == 0:
    run_figures()

print("\n=== Done ===")
