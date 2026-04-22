"""
Example script for the SaturatingLotkaVolterra package.

Generates two simulated datasets from the same SLV model:

  1. Subcommunities (presence/absence)
       Each sample contains a random subset of species at equal OD. Useful for
       species-dropout / knockout style experiments. The number of distinct
       initial conditions is bounded by the number of presence patterns
       (2^N - 1), so derivative values are heavily duplicated across samples.

  2. Full community (Dirichlet alpha = 1)
       Every species is present at a strictly positive OD, with per-species
       ratios drawn uniformly from the simplex and scaled to a fixed total OD.
       This gives one distinct initial condition per sample and therefore
       trajectories / derivatives that cover a much wider region of state
       space; preferable for system identification.

No observational noise is added to either trajectories or analytic derivatives
in either dataset. If noisy observations are required, add noise downstream of
this script.
"""

from SaturatingLotkaVolterra import (
    SLV,
    create_dataframe,
    generate_initial_conditions,
)
import numpy as np
import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
# --- Configuration ---
n_species = 6
n_samples = 1500
t_span = (0.0, 48.0)
t_eval = torch.linspace(0.0, 24.0, 10)

species_names = [f"Species_{i+1}" for i in range(n_species)]

# Sign matrix: -1 = inhibitory (red), +1 = facilitative (blue)
A_sign = torch.tensor([
    [ 0, -1, -1,  1, -1, -1],
    [-1,  0, -1,  1, -1,  1],
    [-1, -1,  0, -1, -1, -1],
    [-1,  1, -1,  0, -1, -1],
    [ 1, -1,  1, -1,  0, -1],
    [-1, -1, -1, -1, -1,  0],
], dtype=torch.float32)

# K2 mask: a deliberate mix of non-monotonic (1, Holling IV, finite K2) and
# monotonic (0, Holling II, K2 -> inf) responses so the per-pair response_curves
# plot illustrates BOTH shapes. The SLV class default is K2_exist_prob=1.0 --
# i.e., all interactions non-monotonic when no mask is provided.
K2_mask = torch.zeros(n_species, n_species)
K2_mask[0, 3] = 1.0  # sp1 <- sp4 (positive, non-monotone)
K2_mask[1, 3] = 1.0  # sp2 <- sp4 (positive, non-monotone)
K2_mask[4, 2] = 1.0  # sp5 <- sp3 (positive, non-monotone)
K2_mask[2, 5] = 1.0  # sp3 <- sp6 (negative, non-monotone)
K2_mask[3, 0] = 1.0  # sp4 <- sp1 (negative, non-monotone)
K2_mask[5, 4] = 1.0  # sp6 <- sp5 (negative, non-monotone)

slv = SLV(
    n_species=n_species,
    interaction_prob=1.0,
    A_sign=A_sign,
    K2_mask=K2_mask,
)

# --- Save / reload model parameters ---
slv.save("example/slv_params.safetensors")
print("Saved example/slv_params.safetensors")
slv = SLV.load("example/slv_params.safetensors")
print(f"Loaded SLV model with {slv.n_species} species")


def simulate_and_save(x0, treatment_ids, prefix):
    """Solve the SLV ODE, then save trajectories and analytic derivatives.

    Writes <prefix>.csv (states x(t)) and <prefix>_derivatives.csv (dx/dt
    evaluated at each solved state). No observational noise is added.
    """
    solution = slv.solve(x0, t_span, t_eval)
    solution_np = solution.detach().numpy()

    df = create_dataframe(
        species_names=species_names,
        t_eval=t_eval.numpy(),
        solution=solution_np,
        treatment_ids=treatment_ids,
    )
    df.to_csv(f"{prefix}.csv", index=False)
    print(f"Saved {prefix}.csv -- shape: {df.shape}")

    solution_flat = solution.reshape(-1, n_species)
    dxdt_flat = slv(solution_flat)
    dxdt_np = dxdt_flat.detach().numpy().reshape(x0.shape[0], len(t_eval), n_species)
    deriv_names = [f"d{name}_dt" for name in species_names]
    df_deriv = create_dataframe(
        species_names=deriv_names,
        t_eval=t_eval.numpy(),
        solution=dxdt_np,
        treatment_ids=treatment_ids,
    )
    df_deriv.to_csv(f"{prefix}_derivatives.csv", index=False)
    print(f"Saved {prefix}_derivatives.csv -- shape: {df_deriv.shape}")
    return solution_np


def plot_trajectories(solution_np, treatment_ids, title, save_path, n_plot=24):
    t_np = t_eval.numpy()
    n_cols = 4
    n_rows = int(np.ceil(n_plot / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
    axes = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, n_species))
    indices = np.linspace(0, solution_np.shape[0] - 1, n_plot, dtype=int)
    for idx, i in enumerate(indices):
        ax = axes[idx]
        for j in range(n_species):
            ax.plot(t_np, solution_np[i, :, j], color=colors[j], label=species_names[j])
        ax.set_title(str(treatment_ids[i]), fontsize=8)
        ax.set_xlabel("Time")
        if idx % n_cols == 0:
            ax.set_ylabel("Abundance")
    for i in range(n_plot, len(axes)):
        axes[i].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=8, ncol=n_species)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")


# --- Dataset 1: subcommunities (presence/absence, equal OD per present species) ---
x0_sub = generate_initial_conditions(
    samples=n_samples,
    num_species=n_species,
    target_total_OD=0.02,
    alpha=None,
    min_community_size=1,
    max_community_size=n_species,
)
present_mask = (x0_sub > 0).int().numpy()
treatment_ids_sub = np.array([
    "sp_" + "_".join(str(j + 1) for j in range(n_species) if present_mask[i, j])
    for i in range(n_samples)
])
solution_sub = simulate_and_save(x0_sub, treatment_ids_sub, "example/dataset_subcommunities")

# --- Dataset 2: full community (all species present, Dirichlet alpha=1 ratios) ---
x0_full = generate_initial_conditions(
    samples=n_samples,
    num_species=n_species,
    target_total_OD=0.02,
    alpha=1.0,
)
treatment_ids_full = np.array([f"sample_{i:04d}" for i in range(n_samples)])
solution_full = simulate_and_save(x0_full, treatment_ids_full, "example/dataset_full_community")

# --- Plot trajectories for each dataset ---
plot_trajectories(
    solution_sub, treatment_ids_sub,
    title="Subcommunity Dynamics (presence/absence; subset of treatments)",
    save_path="example/treatments_subcommunities.png",
)
plot_trajectories(
    solution_full, treatment_ids_full,
    title="Full-Community Dynamics (Dirichlet alpha=1; subset of samples)",
    save_path="example/treatments_full_community.png",
)

# --- Plot N x N pairwise response curves (shared across datasets) ---
A = slv.A.numpy()
K1 = slv.K1.numpy()
K2 = slv.K2.numpy()
mask = slv.interaction_mask.numpy()

x_range = np.linspace(0, 5, 500)
fig, axes = plt.subplots(n_species, n_species, figsize=(2.5 * n_species, 2.5 * n_species),
                         sharex=True, sharey="row")

for i in range(n_species):
    for j in range(n_species):
        ax = axes[i][j]
        if i == j:
            ax.text(0.5, 0.5, "self", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")
            ax.set_facecolor("#f0f0f0")
        elif mask[i, j] == 0:
            ax.text(0.5, 0.5, "no interaction", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="gray")
            ax.set_facecolor("#f8f8f8")
        else:
            denom = x_range + K1[i, j] + x_range**2 / K2[i, j]
            response = A[i, j] * x_range / denom
            color = "tab:blue" if A[i, j] >= 0 else "tab:red"
            ax.plot(x_range, response, color=color, linewidth=1.5)
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        if i == n_species - 1:
            ax.set_xlabel(f"$x_{{{j+1}}}$")
        if j == 0:
            ax.set_ylabel(f"Effect on sp. {i+1}")
        if i == 0:
            ax.set_title(f"Species {j+1}")

fig.suptitle("Pairwise Functional Response Curves\n"
             r"$A_{ij} \, x_j \,/\, (x_j + K_{1,ij} + x_j^2 / K_{2,ij})$"
             "\n(blue = positive, red = negative)",
             fontsize=13)
fig.tight_layout()
fig.savefig("example/response_curves.png", dpi=150, bbox_inches="tight")
print("Saved example/response_curves.png")
