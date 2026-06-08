import torch


class Config:
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Trial parameters
    N_TRIALS = 500
    T_MAX = 1000
    PERTURBATION_STD = 0.05

    # Metric thresholds
    GRAD_NORM_THRESH = 1e-4
    LAMBDA_MIN_THRESH = -1e-3

    # Optimizer list
    OPTIMIZERS = ["GD_fixed", "Adam", "AdamW", "RMSProp", "AdaGrad", "SGD_momentum"]

    # Learning rates to sweep
    LEARNING_RATES = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]

    # Bootstrap
    N_BOOTSTRAP = 1000

    # Part 2
    DIMENSIONS = [2, 10, 50, 100, 500]
    SADDLE_INDICES_D50 = [1, 12, 25]   # k = 1, n//4, n//2 at d=50
    N_TRIALS_HIGHDIM_SMALL = 200        # d <= 50
    N_TRIALS_HIGHDIM_LARGE = 100        # d > 50
    T_MAX_HIGHDIM = 500
    LANCZOS_K = 6                       # eigsh k parameter for d > 20

    # Part 3
    NN_STEPS = 2000
    NN_BATCH = 64
    NN_SEEDS = [0, 1, 2]
    NN_GRAD_CHECK_INTERVAL = 25
    NN_GRAD_NORM_THRESH = 0.05
    NN_SADDLE_LAMBDA_THRESH = -0.01
    NN_SUB_TRIALS = 50
    NN_SUB_T_MAX = 300
    NN_PERTURB_STD = 0.01


def apply_fast(cfg_class):
    cfg_class.N_TRIALS = 50
    cfg_class.T_MAX = 300
    cfg_class.DIMENSIONS = [2, 10]
    cfg_class.NN_STEPS = 200
    cfg_class.LEARNING_RATES = [0.01, 0.1]
    cfg_class.N_BOOTSTRAP = 200
    cfg_class.N_TRIALS_HIGHDIM_SMALL = 50
    cfg_class.N_TRIALS_HIGHDIM_LARGE = 50
    cfg_class.T_MAX_HIGHDIM = 200
    cfg_class.NN_SUB_TRIALS = 10
    cfg_class.NN_SUB_T_MAX = 100
