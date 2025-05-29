from src.data_preprocessing import load_data, preprocess_data, save_processed_data

# Load dataset
df = load_data("data/raw/healthcare_dataset.csv")

# Preprocess using the correct target
processed = preprocess_data(df, target_variable="readmission")

# Save aligned train/test splits
save_processed_data(processed)
