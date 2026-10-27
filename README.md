
# MLflow Experiment Tracking and Data Drift Analysis

## Project Overview
This project demonstrates machine learning experiment tracking using MLflow and data drift analysis using Evidently on the Iris dataset.

## Objectives
- Train a machine learning classification model.
- Track model parameters and evaluation metrics using MLflow.
- Evaluate accuracy, precision, recall, and F1-score.
- Analyze data quality and data drift.
- Generate reports for analysis.

## Technologies Used
- Python
- MLflow
- Scikit-learn
- Pandas
- NumPy
- Evidently

## Project Structure
```text
mlflow-drift-analysis/
├── data/
├── reports/
├── train.py
├── drift_analysis.py
├── classification_report.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository and open its directory:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd mlflow-drift-analysis
```

Create and activate a virtual environment on Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

Train the model and log the experiment:

```bash
python train.py
```

Run data drift analysis:

```bash
python drift_analysis.py
```

Launch the MLflow dashboard:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://127.0.0.1:5000 in your browser.

## Training Results

The recorded model accuracy was 93.33%.

Check the generated reports for the data drift analysis results.

