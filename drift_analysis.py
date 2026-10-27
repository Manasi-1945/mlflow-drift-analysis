
import os
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset, DataSummaryPreset

# Create output directories
os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Load Iris features
iris = load_iris(as_frame=True)
df = iris.data.copy()

# Split into reference and current batches
reference_data, current_data = train_test_split(
    df,
    test_size=0.4,
    random_state=42
)

reference_data = reference_data.reset_index(drop=True).copy()
current_data = current_data.reset_index(drop=True).copy()

# Simulate data drift in two features
current_data["sepal length (cm)"] += 1.0
current_data["petal length (cm)"] += 0.8

# Save datasets for reproducibility
reference_data.to_csv(
    "data/reference_data.csv", index=False
)
current_data.to_csv(
    "data/current_data.csv", index=False
)

print("Reference shape:", reference_data.shape)
print("Current shape:", current_data.shape)

# Define the numerical feature schema
schema = DataDefinition(
    numerical_columns=list(df.columns)
)

reference_dataset = Dataset.from_pandas(
    reference_data,
    data_definition=schema
)

current_dataset = Dataset.from_pandas(
    current_data,
    data_definition=schema
)

# Generate data quality summary
quality_report = Report([
    DataSummaryPreset()
])

quality_result = quality_report.run(
    current_dataset,
    None
)

quality_result.save_html(
    "reports/data_quality_report.html"
)

# Compare reference and current data
drift_report = Report([
    DataDriftPreset()
])

drift_result = drift_report.run(
    current_dataset,
    reference_dataset
)

drift_result.save_html(
    "reports/data_drift_report.html"
)

print("Data quality report saved.")
print("Data drift report saved.")
print("Open the HTML reports in your browser.")
