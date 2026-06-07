# Saddle Escape Efficiency (SEE)

**"Saddle Escape Efficiency: A Novel Metric to Benchmark Learning Rates in Non-Convex Optimization"**

SEE = P_esc / τ_avg, where P_esc is the fraction of trials escaping a saddle region and τ_avg is the mean iterations to escape (conditioned on successful escapes only).

---

## Installation

```bash
cd ~/Desktop/SEE_experiments
python -m venv see_env
source see_env/bin/activate
pip install torch torchvision numpy scipy scikit-learn matplotlib pandas pyyaml joblib pingouin datasets transformers
```

---

## Quick test (< 5 minutes)

```bash
source see_env/bin/activate
python run_all.py --fast
```

This runs with N=50 trials, T_max=200, d_max=20, GPT_steps=500 and should complete in under 5 minutes on Apple M2.

---

## Full run

```bash
# All parts sequentially
python run_all.py

# Individual parts
python run_all.py --part 1
python run_all.py --part 1 2
python run_all.py --part 3 4 5

# Resume Part 1 after interruption
python run_all.py --part 1 --resume

# Regenerate all figures from existing CSVs
python run_all.py --plots-only
```

---

## Expected runtime (Apple M2, 8GB RAM)

| Part | Description                     | Full run  | Fast mode |
|------|---------------------------------|-----------|-----------|
| 1    | Classical 2D experiments        | ~45 min   | ~1 min    |
| 2    | High-dimensional scaling        | ~60 min   | ~1 min    |
| 3    | Neural network (real datasets)  | ~30 min   | ~2 min    |
| 4    | LLM proxy (Tiny GPT)            | ~90 min   | ~1 min    |
| 5    | Synthesis / correlations        | <1 min    | <1 min    |

Total full run: ~4 hours. Fast mode: < 5 minutes.

---

## Expected outputs

```
results/
├── env_log.txt               # Package versions, device info
├── best_lrs.yaml             # Best learning rate per optimizer (from Part 1)
├── part1_results.csv         # SEE metrics for 2D benchmark functions
├── part2_results.csv         # SEE metrics vs dimension
├── part3_results.csv         # NN training metrics + SEE_NN at saddles
├── part4_results.csv         # LLM training metrics + plateau events
├── part5_results.csv         # Correlations: SEE vs training efficiency
└── figures/
    ├── figure1_see_heatmaps.pdf       # Optimizer × LR heatmaps
    ├── figure2_dimension_scaling.pdf  # SEE vs dimension (KEY)
    ├── figure3_saddle_index.pdf       # SEE vs saddle index k
    ├── figure4_nn_training.pdf        # Loss curves + grad norm
    ├── figure5_main_result.pdf        # SEE predicts efficiency (MAIN)
    ├── figure6_escape_types.pdf       # Escape breakdown per optimizer
    └── figure7_escape_time_violins.pdf # τ distributions
```

---

## Reproducing individual figures

```bash
# From project root with venv active:
python -m src.plots.figure1_heatmaps
python -m src.plots.figure5_main_result
# etc.
```

Or use the plots-only flag:
```bash
python run_all.py --plots-only
```

---

## Code structure

```
config.py                  # ALL hyperparameters — no magic numbers elsewhere
run_all.py                 # Master runner
src/
  functions/
    classical_2d.py        # 6 benchmark functions (autograd-compatible)
    nd_functions.py        # High-dim + synthetic saddle
    saddle_finder.py       # Grid search + Newton refinement
  optimizers/
    wrapper.py             # Unified factory for all 7 optimizers
  metrics/
    see.py                 # SEE computation + known limitations docstring
    bootstrap.py           # Vectorized bootstrap CIs
    statistics.py          # Wilcoxon, Bonferroni, Cohen's d
    hessian.py             # Full Hessian (d≤20) + Lanczos (d>20)
  experiments/
    part1_classical.py
    part2_highdim.py
    part3_nn.py            # Real datasets only (Moons, MNIST, Housing)
    part4_llm_proxy.py     # Tiny GPT on Wikitext-2
    part5_synthesis.py     # Predictive validity
  models/
    mlp.py                 # 2-layer MLP, tanh activations
    tiny_gpt.py            # Manual GPT (no HuggingFace model classes)
  data/
    loaders.py             # All dataset loading
  plots/
    figure{1..7}_*.py      # One file per figure
    style.py               # Publication rcParams
  utils/
    seeding.py             # set_all_seeds(seed)
    logging_utils.py       # CSV append, env logging
    parallel.py            # joblib wrappers
    checkpointing.py       # Save/resume via pickle
```

---

## SEE metric

```
SEE_basic:    P_esc = (local_min + diverge) / N,  τ over all escaping trials
SEE_quality:  P_esc = local_min / N,              τ over quality escapes only  ← PRIMARY
SEE_diverge:  diverge / N                                                       ← diagnostic
```

Escape outcomes:
- **LOCAL_MIN**: `||∇f||₂ < 1e-4` AND `min_eigenvalue(H) > -1e-3`
- **DIVERGE**: `||x||₂ > 10·||saddle||₂ + 50`
- **STUCK**: neither within T_max iterations

---

## Known limitations

See `src/metrics/see.py` docstring for the complete list. Key points:

1. Hessian is O(p²) — infeasible for p > 100K. Parts 3 & 4 use gradient norm plateau as a proxy.
2. SEE is local — results are specific to the tested saddle initializations.
3. τ_avg is unreliable when P_esc < 0.1 (flagged in CSV as `reliable=False`).
4. Part 4 uses gradient norm plateau detection, NOT true Hessian-based saddle detection.

---

## Citation

```
@article{see2025,
  title={Saddle Escape Efficiency: A Novel Metric to Benchmark Learning Rates in Non-Convex Optimization},
  year={2025}
}
```
