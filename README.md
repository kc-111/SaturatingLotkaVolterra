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

## Citations:

1. Zenari, Marco, et al. "Generalized Lotka-Volterra systems with quenched random interactions and saturating nonlinear response." Physical Review E 113.2 (2026): 024206.
2. Xiao, Dongmei, and Shigui Ruan. "Global analysis in a predator-prey system with nonmonotonic functional response." SIAM Journal on Applied Mathematics 61.4 (2001): 1445-1472.
3. Tsitouras, Ch. "Runge–Kutta pairs of order 5 (4) satisfying only the first column simplifying assumption." Computers & mathematics with applications 62.2 (2011): 770-775.
