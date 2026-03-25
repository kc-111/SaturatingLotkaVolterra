from SaturatingLotkaVolterra import PiecewiseSkewLinear
import torch
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
n_species = 4
n_samples = 12
T = 1000.0
block_t = 10.0
n_pieces = 5

model = PiecewiseSkewLinear(
    n_species=n_species,
    n_pieces=n_pieces,
    block_t=block_t,
    scale=0.075,
    random_taus=True,
    dirichlet_alpha=1.0,  # uniform on simplex; <1 = sparse, >1 = concentrated
)

# --- Initial conditions on the unit sphere ---
x0 = torch.randn(n_samples, n_species)
x0 = x0 / x0.norm(dim=1, keepdim=True)

t_eval = torch.linspace(0, T, 500)
solution = model.solve(x0, (0.0, T), t_eval)
sol_np = solution.detach().numpy()
t_np = t_eval.numpy()

# --- Plot time series per sample ---
n_cols = 4
n_rows = int(np.ceil(n_samples / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
axes = axes.flatten()

colors = plt.cm.tab10(np.linspace(0, 1, n_species))
species_names = [f"$x_{{{i+1}}}$" for i in range(n_species)]

# Per-block tau schedule from the model (populated after solve)
block_taus_np = model._block_taus.numpy()  # (n_blocks, n_pieces)

for i in range(n_samples):
    ax = axes[i]
    for j in range(n_species):
        ax.plot(t_np, sol_np[i, :, j], color=colors[j], label=species_names[j])
    # Shade blocks and mark per-block phase boundaries
    for b in range(block_taus_np.shape[0]):
        t_block_start = b * block_t
        boundaries = np.cumsum(block_taus_np[b] * block_t)[:-1]
        for pb in boundaries:
            ax.axvline(t_block_start + pb, color="grey", linewidth=0.4, linestyle=":")
        if b % 2 == 1:
            ax.axvspan(t_block_start, t_block_start + block_t, alpha=0.04, color="black")
    ax.set_title(f"Sample {i+1}", fontsize=9)
    ax.set_xlabel("Time")
    if i % n_cols == 0:
        ax.set_ylabel("State")

for i in range(n_samples, len(axes)):
    axes[i].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", fontsize=8, ncol=n_species)
fig.suptitle(
    f"Piecewise Skew-Symmetric Linear System\n"
    f"{n_pieces} pieces, block = {block_t}, "
    + r"random $\tau$ ~ Dirichlet($\alpha$=1)",
    fontsize=13,
)
fig.tight_layout()
fig.savefig("example/piecewise_skew_trajectories.png", dpi=150, bbox_inches="tight")
print("Saved example/piecewise_skew_trajectories.png")

# --- Norm preservation check ---
norms = np.linalg.norm(sol_np, axis=2)  # (n_samples, n_eval)
fig2, ax2 = plt.subplots(figsize=(8, 3))
for i in range(n_samples):
    ax2.plot(t_np, norms[i], alpha=0.5)
ax2.set_xlabel("Time")
ax2.set_ylabel("$\\|x(t)\\|_2$")
ax2.set_title("Norm Preservation (should be constant = 1)")
fig2.tight_layout()
fig2.savefig("example/piecewise_skew_norms.png", dpi=150, bbox_inches="tight")
print("Saved example/piecewise_skew_norms.png")
