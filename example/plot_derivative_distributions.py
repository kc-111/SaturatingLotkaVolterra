"""
Visualize the distribution of time derivatives per species across the two
example datasets (subcommunities vs full community).

Run example/example.py first to generate the CSVs read by this script.
Writes example/derivative_distributions.png.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

datasets = {
    "Subcommunities": HERE / "dataset_subcommunities_derivatives.csv",
    "Full community": HERE / "dataset_full_community_derivatives.csv",
}
loaded = {name: pd.read_csv(path) for name, path in datasets.items()}

species_cols = [c for c in next(iter(loaded.values())).columns if c.startswith("dSpecies_")]
n_species = len(species_cols)

colors = {"Subcommunities": "tab:orange", "Full community": "tab:blue"}

n_cols = 3
n_rows = int(np.ceil(n_species / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.2 * n_rows))
axes = axes.flatten()

for idx, col in enumerate(species_cols):
    ax = axes[idx]
    all_vals = np.concatenate([df[col].values for df in loaded.values()])
    lo, hi = np.quantile(all_vals, [0.001, 0.999])
    if lo == hi:
        lo, hi = float(all_vals.min()), float(all_vals.max())
        if lo == hi:
            lo, hi = lo - 1e-6, hi + 1e-6
    bins = np.linspace(lo, hi, 80)

    for name, df in loaded.items():
        v = df[col].values
        ax.hist(
            v,
            bins=bins,
            histtype="step",
            linewidth=1.6,
            color=colors[name],
            label=name,
        )

    ax.axvline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_title(col)
    ax.set_xlabel("dx/dt")
    ax.set_ylabel("count")
    ax.set_yscale("log")

for i in range(n_species, len(axes)):
    axes[i].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", fontsize=10, bbox_to_anchor=(1.0, 1.0))
fig.suptitle(
    "Per-species distribution of time derivatives\n"
    "(15000 rows per dataset; x clipped to central 99.8% quantile; log-count y)",
    fontsize=13,
)
fig.tight_layout()

out = HERE / "derivative_distributions.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
