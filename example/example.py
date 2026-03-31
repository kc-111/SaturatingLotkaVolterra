from SaturatingLotkaVolterra import SLV, create_dataframe, generate_presence_absence_initial_conditions
import torch
import numpy as np

# --- Configuration ---
n_species = 6
n_samples = 1500
t_span = (0.0, 24.0)
t_eval = torch.linspace(0.0, 24.0, 10)

species_names = [f"Species_{i+1}" for i in range(n_species)]

# Sign matrix: -1 = inhibitory (red), +1 = facilitative (blue)
# sp1: 4 neg / 1 pos,  sp2: 3 neg / 2 pos,  sp3: 5 neg / 0 pos,
# sp4: 4 neg / 1 pos,  sp5: 3 neg / 2 pos,  sp6: 5 neg / 0 pos
A_sign = torch.tensor([
    [ 0, -1, -1,  1, -1, -1],
    [-1,  0, -1,  1, -1,  1],
    [-1, -1,  0, -1, -1, -1],
    [-1,  1, -1,  0, -1, -1],
    [ 1, -1,  1, -1,  0, -1],
    [-1, -1, -1, -1, -1,  0],
], dtype=torch.float32)

# K2 mask: 1 = non-monotone (finite K2), 0 = monotone (K2 → ∞)
K2_mask = torch.zeros(n_species, n_species)
K2_mask[0, 3] = 1.0  # sp1 ← sp4 (positive, non-monotone)
K2_mask[1, 3] = 1.0  # sp2 ← sp4 (positive, non-monotone)
K2_mask[4, 2] = 1.0  # sp5 ← sp3 (positive, non-monotone)

slv = SLV(
    n_species=n_species,
    interaction_prob=1.0,
    A_sign=A_sign,
    K2_mask=K2_mask,
)

# --- Generate presence/absence initial conditions ---
x0 = generate_presence_absence_initial_conditions(
    samples=n_samples,
    num_species=n_species,
    target_total_OD=0.02,
)

# Assign unique treatment ID per sample based on which species are present
present_mask = (x0 > 0).int().numpy()
treatment_ids = np.array([
    "sp_" + "_".join(str(j+1) for j in range(n_species) if present_mask[i, j])
    for i in range(n_samples)
])
# --- Solve ---
solution = slv.solve(x0, t_span, t_eval)  # (n_samples, n_eval, n_species)
solution_np = solution.detach().numpy()

# --- Build DataFrame and save ---
df = create_dataframe(
    species_names=species_names,
    t_eval=t_eval.numpy(),
    solution=solution_np,
    treatment_ids=treatment_ids,
)

df.to_csv("example/dataset.csv", index=False)
print(f"Saved example/dataset.csv — shape: {df.shape}")

# --- Build derivative dataset: evaluate dx/dt at each solved state ---
# Reshape solution to (n_samples * n_eval, n_species), evaluate, reshape back
solution_flat = solution.reshape(-1, n_species)  # (n_samples * n_eval, n_species)
dxdt_flat = slv(solution_flat)                    # (n_samples * n_eval, n_species)
dxdt_np = dxdt_flat.detach().numpy().reshape(n_samples, len(t_eval), n_species)

deriv_names = [f"d{name}_dt" for name in species_names]
df_deriv = create_dataframe(
    species_names=deriv_names,
    t_eval=t_eval.numpy(),
    solution=dxdt_np,
    treatment_ids=treatment_ids,
)
df_deriv.to_csv("example/dataset_derivatives.csv", index=False)
print(f"Saved example/dataset_derivatives.csv — shape: {df_deriv.shape}")

# --- Plot time series for a subset of samples ---
import matplotlib.pyplot as plt

n_plot = 24
t_np = t_eval.numpy()
n_cols = 4
n_rows = int(np.ceil(n_plot / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
axes = axes.flatten()

colors = plt.cm.tab10(np.linspace(0, 1, n_species))

plot_indices = np.linspace(0, n_samples - 1, n_plot, dtype=int)
for idx, i in enumerate(plot_indices):
    ax = axes[idx]
    for j in range(n_species):
        ax.plot(t_np, solution_np[i, :, j], color=colors[j], label=species_names[j])
    ax.set_title(treatment_ids[i], fontsize=9)
    ax.set_xlabel("Time")
    if idx % n_cols == 0:
        ax.set_ylabel("Abundance")

# Hide unused axes
for i in range(n_plot, len(axes)):
    axes[i].set_visible(False)

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", fontsize=8, ncol=n_species)
fig.suptitle("Species Dynamics (subset of treatments)", fontsize=14)
fig.tight_layout()
fig.savefig("example/treatments.png", dpi=150, bbox_inches="tight")
print("Saved example/treatments.png")

# --- Plot N x N pairwise response curves ---
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
