# Industrial Predictive Maintenance

## Recruiter-Focused Project Overview

This project is a full machine learning application for predicting truck air pressure system failures from industrial sensor data. It began as my BSc thesis work and grew into a portfolio-ready product: a trained predictive model, a Streamlit web interface, user authentication, prediction history, explainable AI reports, and a tested Python codebase.

The goal is practical: help maintenance teams identify machines that are likely to fail before the failure becomes expensive downtime. Instead of only training a model in a notebook, I built the surrounding application layer that turns model output into something an operator can upload, inspect, save, and act on.

## What I Built

- A Streamlit web application for batch scoring industrial sensor CSV files.
- A complete preprocessing and inference pipeline using saved model artifacts.
- User registration and login backed by SQLite.
- Persistent prediction history so each account can review previous runs.
- Downloadable prediction results for maintenance workflows.
- Risk labels that translate model probabilities into "Healthy", "Warning", and "Critical" statuses.
- Global and local SHAP explainability reports to show which sensor readings influence failure risk.
- Unit tests for authentication, preprocessing, prediction labeling, and saved-run persistence.
- Model comparison artifacts covering XGBoost, HistGradientBoosting, ExtraTrees, Random Forest, TensorFlow, Logistic Regression, and a dummy baseline.

## Why This Project Matters

Predictive maintenance is a real business problem: unplanned equipment failures can interrupt operations, increase repair costs, and reduce trust in automated systems. This project focuses on the air pressure system failure dataset, where failures are rare but important. That makes the task harder than a simple classification demo because the model must deal with imbalanced data and still catch high-risk cases.

I selected average precision as the main model-selection metric because it is better suited to imbalanced classification than accuracy alone. The final selected model was XGBoost, which achieved:

| Metric | Test Result |
| --- | ---: |
| Accuracy | 99.19% |
| Precision | 79.57% |
| Recall | 88.27% |
| F1 Score | 83.69% |
| ROC AUC | 99.60% |
| Average Precision | 93.02% |

These results show that the model can identify a large share of failure cases while keeping false alarms controlled enough to be useful in a maintenance setting.

## Product Experience

The application is designed like a small operational tool rather than a notebook-only experiment.

1. A user creates an account or logs in.
2. The user uploads a CSV file containing machine sensor readings.
3. The app validates the required sensor columns and applies the saved preprocessing pipeline.
4. The trained model returns failure probabilities for each machine.
5. The app assigns maintenance statuses and displays a batch prediction table.
6. The user can download processed results as a CSV.
7. The app stores the run in SQLite so previous analyses can be reviewed later.
8. SHAP charts explain both global model behavior and the top risk drivers for a selected machine.

This demonstrates the part of machine learning that employers often care about most: taking a model from experimentation into a usable workflow.

## Technical Highlights

- **Machine learning:** XGBoost model selected through validation, cross-validation, and final test evaluation.
- **Data preparation:** Saved preprocessing artifact for consistent feature ordering, numeric conversion, imputation, and scaling.
- **Explainability:** SHAP integration for global feature importance and individual machine-level risk analysis.
- **Application development:** Streamlit frontend with authenticated user sessions and CSV upload/download flows.
- **Persistence:** SQLite database schema for users and historical prediction runs.
- **Security basics:** Password hashing with PBKDF2 and salted hashes.
- **Testing:** Python unittest coverage for core helper functions and database persistence behavior.
- **Reproducibility:** Requirements file, model artifacts, notebooks, metrics, and test suite included in the repository.

## My Role

I designed and implemented the project end to end:

- explored and cleaned the APS sensor dataset;
- compared multiple classification approaches;
- selected the best model using imbalanced-classification metrics;
- saved the trained model and preprocessing pipeline as reusable artifacts;
- built the Streamlit interface around the trained model;
- added authentication, run storage, and result download features;
- integrated SHAP explanations so predictions are easier to trust;
- wrote unit tests for the most important non-UI logic;
- documented setup steps and expected behavior for running the project locally.

## Repository Structure

```text
src/main.py        Streamlit app, authentication, preprocessing, inference, SHAP, persistence
tests/             Unit tests for helper logic and database behavior
models/            Trained model, preprocessing artifact, metrics, and model reports
data/              APS dataset and local SQLite app database
notebooks/         EDA, data cleaning, and model selection experiments
thesis_doc/        Thesis source and supporting figures
README.md          Quick-start technical instructions
```

## How To Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py"
python -m streamlit run src/main.py
```

Then open the Streamlit URL, create an account, upload `data/raw/aps_failure_test_set.csv`, and run batch prediction.

## What This Demonstrates

This project shows that I can work across the full applied machine learning lifecycle: data analysis, model training, evaluation, explainability, application development, persistence, testing, and documentation. It is not only a thesis experiment; it is a working prototype that connects machine learning results to a real user workflow.

For a junior data scientist, machine learning engineer, or Python developer role, this project reflects the kind of practical engineering mindset I would bring to a team: build the model, explain the result, test the core logic, and make the work usable by someone else.
