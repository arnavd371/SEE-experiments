"""Central configuration — every hyperparameter lives here. No magic numbers elsewhere."""

import torch

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
CPU_DEVICE = torch.device('cpu')  # forced CPU for Parts 1 & 2 (joblib-safe)

# ── Reproducibility ───────────────────────────────────────────────────────────
GLOBAL_SEED = 42

# ── Escape classification thresholds ─────────────────────────────────────────
GRAD_TOL = 1e-4           # ||∇f|| < this → local min candidate
EIGEN_TOL_POS = 1e-3      # min eigenvalue > -this → confirmed local min
HESSIAN_DIM_THRESHOLD = 20  # full Hessian for d <= this; Lanczos above

# ── Part 1 — Classical 2D ─────────────────────────────────────────────────────
N_TRIALS_PART1 = 500
T_MAX = 2000
PERTURBATION_STD = 0.05
GRID_SIZE = 200
GRAD_CANDIDATES_TOL = 0.5   # ||∇f|| < this on grid → saddle candidate
SADDLE_DEDUP_RADIUS = 0.5

FUNCTION_DOMAINS = {
    'Himmelblau':       [(-6, 6), (-6, 6)],
    'Rosenbrock':       [(-2, 2), (-1, 3)],
    'Ackley':           [(-5, 5), (-5, 5)],
    'Rastrigin':        [(-5.12, 5.12), (-5.12, 5.12)],
    'Styblinski-Tang':  [(-5, 5), (-5, 5)],
    'Levy':             [(-10, 10), (-10, 10)],
}
FUNCTION_NAMES_2D = list(FUNCTION_DOMAINS.keys())

# ── Part 2 — High-dimensional ─────────────────────────────────────────────────
N_TRIALS_PART2 = 500
N_TRIALS_PART2_HIGHD = 200   # for d > 50
T_MAX_HIGHD = 1000            # for d > 50
DIMENSIONS = [2, 5, 10, 20, 50, 100, 200, 500]
ND_FUNCTION_NAMES = ['Rastrigin-nD', 'Styblinski-nD', 'Ackley-nD', 'Synthetic-Saddle']

# ── Learning rates ─────────────────────────────────────────────────────────────
LEARNING_RATES = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]

# ── Optimizers ─────────────────────────────────────────────────────────────────
OPTIMIZER_NAMES = ['GD_fixed', 'Adam', 'AdamW', 'RMSProp', 'AdaGrad', 'SGD_mom', 'SGD_nesterov']

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8
ADAMW_WD = 0.01
RMSPROP_ALPHA = 0.99
RMSPROP_EPS = 1e-8
ADAGRAD_EPS = 1e-8
SGD_MOMENTUM = 0.9

# ── Bootstrap ─────────────────────────────────────────────────────────────────
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_CI_LO = 2.5
BOOTSTRAP_CI_HI = 97.5
RELIABLE_ESCAPE_MIN = 50   # warn if fewer than this many successful escapes

# ── Lanczos ───────────────────────────────────────────────────────────────────
LANCZOS_K = 6

# ── Part 3 — Neural network ────────────────────────────────────────────────────
NN_RUNS = 5
NN_STEPS = 3000
NN_BATCH_SIZE = 64
NN_SUBTRIAL_N = 100
NN_SUBTRIAL_T_MAX = 500
NN_PERTURBATION_STD = 0.01
NN_SADDLE_CHECK_INTERVAL = 25
NN_SADDLE_GRAD_TOL = 0.05
NN_SADDLE_EIGEN_THRESH = -0.01
NN_LANCZOS_K = 6

MOONS_SAMPLES = 2000
MOONS_NOISE = 0.15
MOONS_RANDOM_STATE = 42
MOONS_CONV_LOSS = 0.05

MNIST_DIGITS = (3, 8)
MNIST_PER_CLASS = 1500
MNIST_CONV_LOSS = 0.1

HOUSING_CONV_LOSS = 0.02

# MLP architecture
MLP_HIDDEN1 = 64
MLP_HIDDEN2 = 32

# ── Part 4 — LLM proxy ────────────────────────────────────────────────────────
LLM_RUNS = 3
LLM_STEPS = 10000
LLM_BATCH_SIZE = 32
LLM_GRAD_CLIP = 1.0
LLM_WARMUP_STEPS = 500
LLM_PLATEAU_CHECK_INTERVAL = 100
LLM_PLATEAU_WINDOW = 50
LLM_PLATEAU_GRAD_TOL = 0.1
LLM_PLATEAU_CONSECUTIVE = 3
LLM_PPL_THRESHOLD = 100
LLM_TRAIN_TOKENS = 500_000
LLM_VAL_TOKENS = 50_000
WIKITEXT_DATASET = 'Salesforce/wikitext'
WIKITEXT_CONFIG = 'wikitext-2-raw-v1'

# Tiny GPT
GPT_N_LAYERS = 4
GPT_N_HEADS = 4
GPT_D_MODEL = 128
GPT_D_FF = 512
GPT_VOCAB_SIZE = 50257
GPT_CONTEXT_LENGTH = 128
GPT_DROPOUT = 0.0

# ── Parallelism ────────────────────────────────────────────────────────────────
N_JOBS = 8

# ── Checkpointing ──────────────────────────────────────────────────────────────
CHECKPOINT_INTERVAL = 10000

# ── Output paths ──────────────────────────────────────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
BEST_LRS_PATH = os.path.join(RESULTS_DIR, 'best_lrs.yaml')

# ── --fast flag overrides ──────────────────────────────────────────────────────
FAST_N_TRIALS = 10
FAST_T_MAX = 200
FAST_D_MAX = 20
FAST_GPT_STEPS = 200
FAST_NN_STEPS = 100
FAST_NN_RUNS = 1
FAST_LLM_RUNS = 1
FAST_BOOTSTRAP = 30
FAST_GRID_SIZE = 40
FAST_MAX_SADDLES = 1           # max saddles to test per function in fast mode
FAST_LRS = [0.001, 0.01, 0.1]  # reduced LR sweep in fast mode
FAST_SADDLE_GRID = 8           # 8×8=64 fsolve starts in fast mode (vs 200×200 full)
FAST_SKIP_SUBTRIAL = True      # skip expensive Hessian sub-trials in Part 3 fast mode
