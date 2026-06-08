import torch

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

GLOBAL_SEED = 42

# Full experiment parameters
N_TRIALS = 500
T_MAX = 1000
PERTURBATION_STD = 0.05
MAX_SADDLES = 3
CURVATURE_CONSTANT = 0.5   # c in r = min(r_max, c / sqrt(|lambda_min| + eps))
CURVATURE_EPSILON = 1e-6
GRAD_TOL = 1e-4             # gradient norm threshold for local-min check
EIGEN_TOL_MIN = 1e-3        # lambda_min threshold for true local min vs saddle
LEARNING_RATES = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
BOOTSTRAP_RESAMPLES = 1000
NN_RUNS = 3
NN_STEPS = 2000
NN_BATCH_SIZE = 64
SENSITIVITY_CONSTANTS = [0.25, 0.5, 1.0, 2.0]
DIMENSIONS = [2, 10, 50, 100, 500]
NN_SUBTRIAL_N = 50
NN_SUBTRIAL_T_MAX = 300
NN_CHECK_INTERVAL = 25
NN_GRAD_TOL = 0.05
NN_LAMBDA_TOL = 0.01

# --fast flag overrides
FAST_N_TRIALS = 20
FAST_T_MAX = 100
FAST_DIMENSIONS = [2, 10]
FAST_NN_STEPS = 100
FAST_NN_RUNS = 1

# Saddle-finding grid resolution
SADDLE_GRID = 200
SADDLE_GRAD_THRESH = 0.5
SADDLE_DEDUP_RADIUS = 0.5

# Optimizers
OPTIMIZERS = {
    'GD_fixed':     {'type': 'SGD',      'kwargs': {}},
    'Adam':         {'type': 'Adam',     'kwargs': {}},
    'AdamW':        {'type': 'AdamW',    'kwargs': {'weight_decay': 0.01}},
    'RMSProp':      {'type': 'RMSprop', 'kwargs': {}},
    'AdaGrad':      {'type': 'Adagrad', 'kwargs': {}},
    'SGD_momentum': {'type': 'SGD',     'kwargs': {'momentum': 0.9}},
}
