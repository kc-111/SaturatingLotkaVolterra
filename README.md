# SaturatingLotkaVolterra
General Lotka Volterra with saturating responses and nonmonotonic interactions.

## Installation

```bash
# Create and activate environment
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

# CPU only
pip install -e .

# GPU (CUDA 12.6)
pip install -e .[gpu] --extra-index-url https://download.pytorch.org/whl/cu126
```

## Example

`example/example.py` simulates two datasets from the same SLV model:

- `dataset_subcommunities.csv` / `dataset_subcommunities_derivatives.csv`
  — presence/absence initial conditions (random species subsets at equal OD).
- `dataset_full_community.csv` / `dataset_full_community_derivatives.csv`
  — all species present, per-species ratios drawn from `Dirichlet(alpha=1)`
  and scaled to a fixed total OD.

Both datasets are **noise-free**: trajectories are the raw ODE solutions and
derivatives are evaluated analytically from the SLV model. Add observational
noise downstream if required.

Initial conditions come from a single unified sampler,
`SaturatingLotkaVolterra.generate_initial_conditions`. Key knobs:
`alpha` (Dirichlet concentration for per-species ratios, or `None` for an
equal split), and `min_community_size` / `max_community_size` (community-size
range; default = `num_species`, i.e. all species present).

## Citations:

1. Zenari, Marco, et al. "Generalized Lotka-Volterra systems with quenched random interactions and saturating nonlinear response." Physical Review E 113.2 (2026): 024206.
2. Xiao, Dongmei, and Shigui Ruan. "Global analysis in a predator-prey system with nonmonotonic functional response." SIAM Journal on Applied Mathematics 61.4 (2001): 1445-1472.
3. Tsitouras, Ch. "Runge–Kutta pairs of order 5 (4) satisfying only the first column simplifying assumption." Computers & mathematics with applications 62.2 (2011): 770-775.
