import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from collections import Counter

def train_models(X_train, y_train, cv=5, random_state=42):
    """
    Train multiple models on the training data
    
    Args:
        X_train: Preprocessed training features
        y_train: Training target variable
        cv: Number of cross-validation folds
        random_state: Random seed for reproducibility
        
    Returns:
        dict with trained models and their cross-validation scores
    """
    # Initialize models
    # Calculate class ratio for XGBoost scale_pos_weight
    class_counts = Counter(y_train)
    neg, pos = class_counts[0], class_counts[1]
    scale_pos_weight = neg / pos if pos != 0 else 1.0

    models = {
        'logistic_regression': LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver='liblinear',
            class_weight='balanced'  # ⚠️ KEY ADDITION
        ),
        'random_forest': RandomForestClassifier(
            random_state=random_state,
            class_weight='balanced'  # ⚠️ KEY ADDITION
        ),
        'xgboost': XGBClassifier(
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight  # ⚠️ KEY ADDITION
        )
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        
        # Train model on full training set
        model.fit(X_train, y_train)
        
        # Store results
        results[name] = {
            'model': model,
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores)
        }
        
        print(f"{name} - Mean ROC AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    return results

def hyperparameter_tuning(X_train, y_train, cv=5, random_state=42):
    """
    Perform hyperparameter tuning for each model
    
    Args:
        X_train: Preprocessed training features
        y_train: Training target variable
        cv: Number of cross-validation folds
        random_state: Random seed for reproducibility
        
    Returns:
        dict with tuned models
    """
    # Parameter grids
    param_grids = {
        'logistic_regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'class_weight': [None, 'balanced'],
            'solver': ['liblinear', 'saga']
        },
        'random_forest': {
            "n_estimators": [100, 200, 500],
             "max_depth": [3, 6, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': [None, 'balanced']
        },
        'xgboost': {
            "n_estimators": [100, 200, 500],
             "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0],
            "scale_pos_weight": [1, 2, 5]
        }
    }
    
    # Base models
    base_models = {
        'logistic_regression': LogisticRegression(random_state=random_state, max_iter=1000),
        'random_forest': RandomForestClassifier(random_state=random_state),
        'xgboost': XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    }
    
    # Train and tune each model
    tuned_models = {}
    for name, model in base_models.items():
        print(f"Tuning {name}...")
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grids[name], cv=cv, scoring='roc_auc', 
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        
        # Store results
        tuned_models[name] = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
        
        print(f"{name} - Best ROC AUC: {grid_search.best_score_:.4f}")
        print(f"Best parameters: {grid_search.best_params_}")
    
    return tuned_models

def evaluate_model(model, X_test, y_test, model_name='Model'):
    """
    Evaluate model performance on test data
    
    Args:
        model: Trained model
        X_test: Preprocessed test features
        y_test: Test target variable
        model_name: Name of the model
        
    Returns:
        dict with evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'average_precision': average_precision_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    # Print results
    print(f"\n{model_name} Evaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    return metrics

def plot_roc_curve(models_dict, X_test, y_test, save_path=None):
    """
    Plot ROC curves for multiple models
    
    Args:
        models_dict: Dict with model names and fitted models
        X_test: Preprocessed test features
        y_test: Test target variable
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    for name, model_info in models_dict.items():
        model = model_info['model']
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Add labels and legend
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
    
    plt.tight_layout()
    
    return plt

def plot_precision_recall_curve(models_dict, X_test, y_test, save_path=None):
    """
    Plot precision-recall curves for multiple models
    
    Args:
        models_dict: Dict with model names and fitted models
        X_test: Preprocessed test features
        y_test: Test target variable
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    for name, model_info in models_dict.items():
        model = model_info['model']
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        plt.plot(recall, precision, lw=2, label=f'{name} (AP = {avg_precision:.3f})')
    
    # Add labels and legend
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    
    if save_path:
        plt.savefig(save_path)
    
    plt.tight_layout()
    
    return plt

def get_feature_importance(model, feature_names, model_type='tree', top_n=20):
    """
    Get feature importance from the model
    
    Args:
        model: Trained model
        feature_names: List of feature names
        model_type: Type of model ('tree' or 'linear')
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature importance
    """
    if model_type == 'tree':
        # For tree-based models (Random Forest, XGBoost)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            raise ValueError("Model does not have feature_importances_ attribute")
    elif model_type == 'linear':
        # For linear models (Logistic Regression)
        if hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            raise ValueError("Model does not have coef_ attribute")
    else:
        raise ValueError("model_type must be 'tree' or 'linear'")
    
    # Create DataFrame
    if feature_names is not None and len(importances) == len(feature_names):
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
    else:
        feature_importance = pd.DataFrame({
            'feature': [f'Feature_{i}' for i in range(len(importances))],
            'importance': importances
        })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Get top N features
    if top_n:
        feature_importance = feature_importance.head(top_n)
    
    return feature_importance

def plot_feature_importance(feature_importance, title='Feature Importance', save_path=None):
    """
    Plot feature importance
    
    Args:
        feature_importance: DataFrame with feature importance
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Plot feature importance
    sns.barplot(x='importance', y='feature', data=feature_importance)
    
    # Add labels and title
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.tight_layout()
    
    return plt

def save_model(model, model_name, output_dir='models'):
    """
    Save trained model to disk
    
    Args:
        model: Trained model
        model_name: Name of the model
        output_dir: Directory to save model
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = f"{output_dir}/{model_name}.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model_path

if __name__ == "__main__":
    # Example usage
    # Load processed data
    X_train = np.load("data/processed/X_train_processed.npy")
    X_test = np.load("data/processed/X_test_processed.npy") 
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")
    
    # Train models
    model_results = train_models(X_train, y_train)
    
    # Evaluate best model
    best_model_name = max(model_results, key=lambda x: model_results[x]['mean_cv_score'])
    best_model = model_results[best_model_name]['model']
    evaluation_metrics = evaluate_model(best_model, X_test, y_test, model_name=best_model_name)
    
    # Save best model
    save_model(best_model, best_model_name)