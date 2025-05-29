import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

def load_data(filepath):
    """
    Load the healthcare dataset
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        pandas DataFrame with the loaded data
    """
    data = pd.read_csv(filepath)
    print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns")
    return data

def explore_data(data):
    """
    Perform basic exploratory data analysis
    
    Args:
        data: pandas DataFrame
        
    Returns:
        dict with EDA results
    """
    # Basic statistics
    eda_results = {
        'shape': data.shape,
        'columns': list(data.columns),
        'dtypes': data.dtypes,
        'missing_values': data.isnull().sum(),
        'missing_percentage': (data.isnull().sum() / len(data) * 100).round(2),
        'descriptive_stats': data.describe(include='all')
    }
    
    # Class distribution for target variables
    if 'readmission' in data.columns:
        eda_results['readmission_distribution'] = data['readmission'].value_counts(normalize=True)
        
    if 'icu_transfer' in data.columns:
        eda_results['icu_transfer_distribution'] = data['icu_transfer'].value_counts(normalize=True)
        
    return eda_results

def preprocess_data(data, target_variable='readmission', test_size=0.2, random_state=42):
    """
    Preprocess the data for machine learning
    
    Args:
        data: pandas DataFrame with raw data
        target_variable: Target variable to predict ('readmission' or 'icu_transfer')
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        dict with processed data and preprocessing objects
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Ensure target variable exists
    if target_variable not in df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in dataset")
    
    # Extract features and target
    y = df[target_variable]
    X = df.drop(columns=[target_variable])
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Fit preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    try:
        # For scikit-learn >= 1.0
        feature_names = numeric_features.copy()
        if categorical_features:
            # Get categorical feature names after one-hot encoding
            cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
            feature_names.extend(cat_feature_names)
    except:
        # For older scikit-learn versions
        feature_names = None
    
    return {
        'X_train': X_train,
        'X_test': X_test, 
        'y_train': y_train,
        'y_test': y_test,
        'X_train_processed': X_train_processed,
        'X_test_processed': X_test_processed,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }

def save_processed_data(processed_data, output_dir='data/processed'):
    """
    Save processed data to disk
    
    Args:
        processed_data: Dict with processed data
        output_dir: Directory to save processed data
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    np.save(f"{output_dir}/X_train_processed.npy", processed_data['X_train_processed'])
    np.save(f"{output_dir}/X_test_processed.npy", processed_data['X_test_processed'])
    np.save(f"{output_dir}/y_train.npy", processed_data['y_train'])
    np.save(f"{output_dir}/y_test.npy", processed_data['y_test'])
    
    # Save original data
    processed_data['X_train'].to_csv(f"{output_dir}/X_train.csv", index=False)
    processed_data['X_test'].to_csv(f"{output_dir}/X_test.csv", index=False)
    
    print(f"Processed data saved to {output_dir}")

if __name__ == "__main__":
    # Example usage
    data = load_data("data/raw/healthcare_dataset.csv")
    eda_results = explore_data(data)
    processed_data = preprocess_data(data, target_variable='readmission')
    save_processed_data(processed_data)