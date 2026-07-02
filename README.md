# Predictive Maintenance on Industrial Sensor Data
For a polished recruiter-facing overview, see [README_RECRUITERS.md](README_RECRUITERS.md).

BSc Thesis Project — Alex Farajallah, ELTE Computer Science, 2026

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate   # Linux/macOS
# or: .venv\Scripts\activate                          # Windows
pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py"
python -m streamlit run src/main.py 
```
Open http://localhost:8501, register an account, log in.
## Test the system
Upload `data/raw/aps_failure_test_set.csv` for an end-to-end smoke test.
Expected: ~333 machines flagged on the full 16,000-row test set.

## Environment
- Python 3.10, 3.11, or 3.12 (NOT 3.13 — joblib artefacts require sklearn 1.7.2)
- 4 GB RAM, 1 GB free disk

## Project structure
- src/main.py      — Streamlit application (auth, DB, inference, SHAP)
- tests/           — unittest suite (6 tests)
- models/          — trained joblib artefacts (XGBoost selected, HGB backup)
- data/            — APS dataset (raw and cleaned splits)
- notebooks/       — EDA, cleaning, model selection
- thesis_doc/      — LaTeX source and figures for the thesis

