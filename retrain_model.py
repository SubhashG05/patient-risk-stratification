from src.data_preprocessing import load_data, preprocess_data, save_processed_data
from src.model_training import train_models, evaluate_model, save_model
import joblib

# Load raw data
data = load_data("data/raw/healthcare_dataset.csv")

# Preprocess with correct target
processed = preprocess_data(data, target_variable="readmission")

# Save aligned test/train data
save_processed_data(processed)

# Train model
models = train_models(processed["X_train_processed"], processed["y_train"])

# Get best model
best_model_name = max(models, key=lambda x: models[x]["mean_cv_score"])
best_model = models[best_model_name]["model"]
print(f"Best model: {best_model_name} with mean CV score: {models[best_model_name]['mean_cv_score']}")

# Save model + preprocessor + feature names
save_model(best_model, "readmission_model")
joblib.dump(processed["preprocessor"], "models/readmission_preprocessor.joblib")
joblib.dump(processed["feature_names"], "models/readmission_feature_names.joblib")

print(f"✅ Model trained and saved with {processed['X_train_processed'].shape[1]} features.")
