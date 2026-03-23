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