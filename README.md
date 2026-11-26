# market-making-sim

Optimal market making research: naive vs Avellaneda–Stoikov, calibrated to BTCUSDT. See CLAUDE.md for the project guide and DECISIONS.md for judgement calls.

## Setup

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e . --no-deps
.venv\Scripts\python -m pytest
```
