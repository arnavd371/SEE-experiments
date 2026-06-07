"""Publication-quality matplotlib style applied to all SEE figures."""

import matplotlib.pyplot as plt
import matplotlib as mpl

PUBLICATION_STYLE = {
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
}

OPTIMIZER_COLORS = {
    'GD_fixed':     '#1f77b4',
    'Adam':         '#ff7f0e',
    'AdamW':        '#2ca02c',
    'RMSProp':      '#d62728',
    'AdaGrad':      '#9467bd',
    'SGD_mom':      '#8c564b',
    'SGD_nesterov': '#e377c2',
}

OUTCOME_COLORS = {
    'local_min': '#2ca02c',
    'diverge':   '#ff7f0e',
    'stuck':     '#d62728',
}


def apply_style():
    plt.rcParams.update(PUBLICATION_STYLE)


def optimizer_color(opt_name: str) -> str:
    return OPTIMIZER_COLORS.get(opt_name, '#333333')


def save_figure(fig, path: str, tight: bool = True):
    """Save figure at 300dpi with tight layout."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')
