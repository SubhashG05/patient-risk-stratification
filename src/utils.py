import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import shap
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.cm as cm

def load_model(model_path):
    """
    Load a saved model
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    return joblib.load(model_path)

def generate_risk_scores(probabilities, num_bins=5):
    """
    Convert probabilities to risk scores
    
    Args:
        probabilities: Array of probability predictions
        num_bins: Number of risk categories
        
    Returns:
        Array of risk scores (1 to num_bins)
    """
    # Create bins
    bins = np.linspace(0, 1, num_bins + 1)
    
    # Convert probabilities to risk scores
    risk_scores = np.digitize(probabilities, bins)
    
    # Adjust to 1-based indexing for better interpretability
    return risk_scores

def generate_risk_labels(risk_scores, num_bins=5):
    """
    Convert risk scores to descriptive labels
    
    Args:
        risk_scores: Array of risk scores
        num_bins: Number of risk categories
        
    Returns:
        Array of risk labels
    """
    # Define risk labels
    if num_bins == 3:
        risk_labels = ['Low', 'Moderate', 'High']
    elif num_bins == 5:
        risk_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    else:
        risk_labels = [f'Level {i+1}' for i in range(num_bins)]
    
    # Map scores to labels
    return np.array([risk_labels[min(score-1, len(risk_labels)-1)] for score in risk_scores])

def plot_risk_distribution(risk_labels, save_path=None):
    """
    Plot distribution of risk categories
    
    Args:
        risk_labels: Array of risk labels
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Count risk categories
    risk_counts = pd.Series(risk_labels).value_counts().sort_index()
    
    # Plot
    ax = sns.barplot(x=risk_counts.index, y=risk_counts.values)
    
    # Add labels
    plt.xlabel('Risk Category')
    plt.ylabel('Number of Patients')
    plt.title('Distribution of Patient Risk Categories')
    
    # Add count labels
    for i, v in enumerate(risk_counts.values):
        ax.text(i, v + 5, str(v), ha='center')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.tight_layout()
    
    return plt

def plot_shap_summary(model, X, feature_names=None, plot_type='bar', max_display=20):
    """
    Generate SHAP summary plots for model interpretability
    
    Args:
        model: Trained model
        X: Feature matrix
        feature_names: List of feature names
        plot_type: Type of SHAP plot ('bar', 'dot', or 'violin')
        max_display: Maximum number of features to display
    """
    # Initialize SHAP explainer based on model type
    if hasattr(model, 'predict_proba'):
        explainer = shap.Explainer(model)
    else:
        explainer = shap.Explainer(model.predict, X)
    
    # Calculate SHAP values
    shap_values = explainer(X)
    
    # Set feature names if provided
    if feature_names is not None:
        shap_values.feature_names = feature_names
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    if plot_type == 'bar':
        shap.plots.bar(shap_values, max_display=max_display, show=False)
    elif plot_type == 'dot':
        shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    elif plot_type == 'violin':
        shap.plots.violin(shap_values, max_display=max_display, show=False)
    else:
        raise ValueError("plot_type must be 'bar', 'dot', or 'violin'")
    
    plt.tight_layout()
    
    return plt

def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if specified
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=labels if labels else ['Negative', 'Positive'],
                yticklabels=labels if labels else ['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.tight_layout()
    
    return plt

def find_optimal_threshold(y_true, y_prob, criterion='balanced'):
    """
    Find the optimal threshold for binary classification
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        criterion: Criterion for threshold selection ('balanced', 'f1', 'precision', 'recall')
        
    Returns:
        Optimal threshold
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    if criterion == 'balanced':
        # Maximize sensitivity + specificity
        optimal_idx = np.argmax(tpr - fpr)
    elif criterion == 'f1':
        # Calculate F1 score for each threshold
        precision = []
        recall = []
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tp = np.sum((y_true == 1) & (y_pred == 1))
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision.append(p)
            recall.append(r)
        
        f1_scores = [2 * p * r / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]
        optimal_idx = np.argmax(f1_scores)
    elif criterion == 'precision':
        # Calculate precision for each threshold
        precision = []
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tp = np.sum((y_true == 1) & (y_pred == 1))
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            precision.append(p)
        
        optimal_idx = np.argmax(precision)
    elif criterion == 'recall':
        # Calculate recall for each threshold
        recall = []
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tp = np.sum((y_true == 1) & (y_pred == 1))
            
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            recall.append(r)
        
        optimal_idx = np.argmax(recall)
    else:
        raise ValueError("criterion must be 'balanced', 'f1', 'precision', or 'recall'")
    
    # Return optimal threshold
    return thresholds[optimal_idx]

def create_sample_healthcare_dataset(n_samples=1000, random_state=42):
    """
    Create a synthetic healthcare dataset for demonstration purposes
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        
    Returns:
        pandas DataFrame with synthetic data
    """
    np.random.seed(random_state)
    
    # Generate data
    data = {
        'age': np.random.normal(65, 15, n_samples).clip(18, 100).astype(int),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'length_of_stay': np.random.poisson(5, n_samples) + 1,
        'num_procedures': np.random.poisson(2, n_samples),
        'num_medications': np.random.poisson(5, n_samples),
        'num_lab_procedures': np.random.poisson(10, n_samples),
        'num_diagnoses': np.random.randint(1, 6, n_samples),
        'blood_pressure_systolic': np.random.normal(130, 20, n_samples).clip(80, 220).astype(int),
        'blood_pressure_diastolic': np.random.normal(80, 10, n_samples).clip(40, 120).astype(int),
        'heart_rate': np.random.normal(80, 15, n_samples).clip(40, 180).astype(int),
        'respiratory_rate': np.random.normal(18, 5, n_samples).clip(8, 40).astype(int),
        'temperature': np.random.normal(98.6, 1, n_samples),
        'oxygen_saturation': np.random.normal(96, 3, n_samples).clip(70, 100).astype(int),
        'primary_diagnosis': np.random.choice([
            'Diabetes', 'Hypertension', 'Heart Failure', 'COPD', 
            'Pneumonia', 'Stroke', 'Kidney Disease', 'Cancer'
        ], n_samples),
        'emergency_admission': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'insurance': np.random.choice(['Medicare', 'Medicaid', 'Private', 'Self-Pay'], n_samples),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_samples),
        'previous_admissions': np.random.poisson(1, n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add comorbidities
    comorbidities = ['Diabetes', 'Hypertension', 'Heart Disease', 'COPD', 'Cancer', 'Obesity']
    for comorbidity in comorbidities:
        df[f'comorbidity_{comorbidity}'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # Create target variables with realistic relationships to features
    
    # Calculate readmission probability
    readmission_prob = 0.1  # Base probability
    
    # Age increases readmission risk
    readmission_prob += 0.003 * (df['age'] - 50).clip(0) / 10
    
    # Previous admissions increase risk
    readmission_prob += 0.05 * df['previous_admissions']
    
    # Number of diagnoses increases risk
    readmission_prob += 0.02 * df['num_diagnoses']
    
    # Comorbidities increase risk
    for comorbidity in comorbidities:
        readmission_prob += 0.03 * df[f'comorbidity_{comorbidity}']
    
    # Emergency admission increases risk
    readmission_prob += 0.1 * df['emergency_admission']
    
    # Length of stay relationship is U-shaped
    length_factor = np.abs(df['length_of_stay'] - 5) / 10
    readmission_prob += 0.05 * length_factor
    
    # Clip probability
    readmission_prob = readmission_prob.clip(0.05, 0.95)
    
    # Generate binary outcome
    df['readmission'] = np.random.binomial(1, readmission_prob)
    
    # Calculate ICU transfer probability (correlated with readmission)
    icu_prob = 0.08  # Base probability
    
    # Correlated with readmission
    icu_prob += 0.2 * df['readmission']
    
    # Vital signs affect ICU transfer risk
    icu_prob += 0.01 * np.abs(df['blood_pressure_systolic'] - 120) / 10
    icu_prob += 0.01 * np.abs(df['heart_rate'] - 80) / 10
    icu_prob += 0.02 * np.abs(df['respiratory_rate'] - 16) / 5
    icu_prob -= 0.02 * (df['oxygen_saturation'] - 90) / 10
    
    # Age increases ICU risk
    icu_prob += 0.002 * (df['age'] - 50).clip(0) / 10
    
    # Number of diagnoses increases risk
    icu_prob += 0.015 * df['num_diagnoses']
    
    # Emergency admission increases risk
    icu_prob += 0.1 * df['emergency_admission']
    
    # Clip probability
    icu_prob = icu_prob.clip(0.02, 0.9)
    
    # Generate binary outcome
    df['icu_transfer'] = np.random.binomial(1, icu_prob)
    
    return df