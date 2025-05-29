import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import shap
import base64
import io
from sklearn.preprocessing import StandardScaler
from PIL import Image
import traceback

# Import from modules
import sys
sys.path.append('.')
from src.data_preprocessing import preprocess_data
from src.utils import (
    load_model, generate_risk_scores, generate_risk_labels,
    plot_risk_distribution, plot_shap_summary, plot_confusion_matrix,
    find_optimal_threshold, create_sample_healthcare_dataset
)

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Set page configuration
st.set_page_config(
    page_title="Patient Risk Stratification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set title
st.title("🏥 Patient Risk Stratification Model")

# Sidebar
st.sidebar.header("Settings")

# Model selection
target_options = {
    'readmission': 'Hospital Readmission',
    'icu_transfer': 'ICU Transfer'
}
target_variable = st.sidebar.selectbox(
    "Select Risk Type to Predict",
    options=list(target_options.keys()),
    format_func=lambda x: target_options[x]
)

# Risk threshold
risk_threshold = st.sidebar.slider(
    "Risk Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Threshold for classifying patients as high risk"
)

# Number of risk categories
num_risk_categories = st.sidebar.selectbox(
    "Number of Risk Categories",
    options=[3, 5],
    index=1,
    help="Number of risk categories for stratification"
)

# Model explanation settings
max_features_display = st.sidebar.slider(
    "Max Features to Display",
    min_value=5,
    max_value=30,
    value=15,
    step=1,
    help="Maximum number of features to display in importance plots"
)

# Check if model exists, if not, train one with sample data
model_path = f"models/{target_variable}_model.joblib"
preprocessor_path = f"models/{target_variable}_preprocessor.joblib"

if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
    st.sidebar.warning("No trained model found. Using a sample model for demonstration.")
    
    # Create sample dataset
    with st.spinner("Generating sample dataset..."):
        sample_df = create_sample_healthcare_dataset(n_samples=2000)
        
        # Save sample dataset
        os.makedirs("data/raw", exist_ok=True)
        sample_df.to_csv("data/raw/healthcare_dataset.csv", index=False)
        
        # Preprocess data
        processed_data = preprocess_data(sample_df, target_variable=target_variable)
        
        # Save preprocessor
        os.makedirs("models", exist_ok=True)
        joblib.dump(processed_data['preprocessor'], preprocessor_path)
        
        # Train a simple model
        from xgboost import XGBClassifier
        model = XGBClassifier(random_state=42)
        model.fit(
            processed_data['X_train_processed'],
            processed_data['y_train']
        )
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save feature names
        feature_names = processed_data['feature_names']
        if feature_names is not None:
            joblib.dump(feature_names, f"models/{target_variable}_feature_names.joblib")
    
    st.sidebar.success("Sample model created successfully!")

# Load model and preprocessor
model = load_model(model_path)
preprocessor = load_model(preprocessor_path)

# Try to load feature names
try:
    feature_names = joblib.load(f"models/{target_variable}_feature_names.joblib")
except:
    feature_names = None

# Main content
st.markdown("""
This application helps healthcare providers identify patients at high risk of hospital readmission or ICU transfer.
By stratifying patients based on their risk levels, hospitals can allocate resources more efficiently and implement
targeted interventions to improve patient outcomes.
""")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Patient Risk Assessment",
    "Model Performance",
    "Model Explanation",
    "About"
])

# Tab 1: Patient Risk Assessment
with tab1:
    st.header("Patient Risk Assessment")
    
    # File upload or manual entry toggle
    input_method = st.radio(
        "Select input method",
        options=["Upload Patient Data CSV", "Enter Patient Data Manually", "Generate Sample Patients"],
        horizontal=True
    )
    
    if input_method == "Upload Patient Data CSV":
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file with patient data", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Read data
                patient_data = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded data with {patient_data.shape[0]} patients and {patient_data.shape[1]} features.")
                
                # Show first few rows
                st.subheader("Patient Data Preview")
                st.dataframe(patient_data.head())
                
                # Process and predict
                if st.button("Calculate Risk Scores"):
                    with st.spinner("Processing data and calculating risk scores..."):
                        # Preprocess data
                        X = preprocessor.transform(patient_data)
                        
                        # Predict probabilities
                        risk_probabilities = model.predict_proba(X)[:, 1]
                        
                        # Generate risk scores and labels
                        risk_scores = generate_risk_scores(risk_probabilities, num_bins=num_risk_categories)
                        risk_labels = generate_risk_labels(risk_scores, num_bins=num_risk_categories)
                        
                        # Add results to dataframe
                        results_df = patient_data.copy()
                        results_df['Risk Probability'] = risk_probabilities
                        results_df['Risk Score'] = risk_scores
                        results_df['Risk Category'] = risk_labels
                        results_df['High Risk'] = risk_probabilities >= risk_threshold
                        
                        # Display results
                        st.subheader("Risk Assessment Results")
                        
                        # Color high risk patients
                        def highlight_high_risk(val):
                            if val == True:
                                return 'background-color: #ffcccc'
                            return ''
                        
                        # Display styled dataframe
                        st.dataframe(
                            results_df.style.applymap(
                                highlight_high_risk, 
                                subset=['High Risk']
                            )
                        )
                        
                        # Plot risk distribution
                        st.subheader("Risk Distribution")
                        fig = plot_risk_distribution(risk_labels)
                        st.pyplot(fig)
                        
                        # Summary statistics
                        st.subheader("Summary Statistics")
                        total_patients = len(results_df)
                        high_risk_count = results_df['High Risk'].sum()
                        high_risk_percentage = high_risk_count / total_patients * 100
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Patients", total_patients)
                        col2.metric("High Risk Patients", high_risk_count)
                        col3.metric("High Risk Percentage", f"{high_risk_percentage:.1f}%")
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="risk_assessment_results.csv">Download Results as CSV</a>'
                        st.markdown(href, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please ensure your CSV file has the required features for the model.")
    
    elif input_method == "Enter Patient Data Manually":
        st.info("Please enter patient information below.")
        
        # Create form for manual entry
        with st.form("patient_form"):
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            # Demographics
            with col1:
                st.subheader("Demographics")
                age = st.slider("Age", 18, 100, 65)
                gender = st.selectbox("Gender", ["Male", "Female"])
                marital_status = st.selectbox(
                    "Marital Status", 
                    ["Single", "Married", "Divorced", "Widowed"]
                )
                insurance = st.selectbox(
                    "Insurance", 
                    ["Medicare", "Medicaid", "Private", "Self-Pay"]
                )
            
            # Clinical data
            with col2:
                st.subheader("Clinical Data")
                length_of_stay = st.slider("Length of Stay (days)", 1, 30, 5)
                num_procedures = st.slider("Number of Procedures", 0, 10, 2)
                num_medications = st.slider("Number of Medications", 0, 20, 5)
                num_lab_procedures = st.slider("Number of Lab Procedures", 0, 30, 10)
                num_diagnoses = st.slider("Number of Diagnoses", 1, 10, 3)
            
            # Vital signs
            st.subheader("Vital Signs")
            col3, col4, col5 = st.columns(3)
            
            with col3:
                systolic_bp = st.slider("Systolic BP (mmHg)", 80, 220, 120)
                diastolic_bp = st.slider("Diastolic BP (mmHg)", 40, 120, 80)
            
            with col4:
                heart_rate = st.slider("Heart Rate (bpm)", 40, 180, 80)
                respiratory_rate = st.slider("Respiratory Rate (breaths/min)", 8, 40, 16)
            
            with col5:
                temperature = st.slider("Temperature (°F)", 95.0, 104.0, 98.6)
                oxygen_saturation = st.slider("Oxygen Saturation (%)", 70, 100, 95)
            
            # Comorbidities
            st.subheader("Comorbidities")
            col6, col7 = st.columns(2)
            
            with col6:
                diabetes = st.checkbox("Diabetes")
                hypertension = st.checkbox("Hypertension")
                heart_disease = st.checkbox("Heart Disease")
            
            with col7:
                copd = st.checkbox("COPD")
                cancer = st.checkbox("Cancer")
                obesity = st.checkbox("Obesity")
            
            # Other information
            st.subheader("Other Information")
            emergency_admission = st.checkbox("Emergency Admission")
            primary_diagnosis = st.selectbox(
                "Primary Diagnosis",
                ["Diabetes", "Hypertension", "Heart Failure", "COPD", 
                 "Pneumonia", "Stroke", "Kidney Disease", "Cancer"]
            )
            previous_admissions = st.slider("Previous Admissions", 0, 10, 1)
            
            # Submit button
            submitted = st.form_submit_button("Calculate Risk")
        
        if submitted:
            # Create a DataFrame from form inputs
            patient_data = pd.DataFrame({
                'age': [age],
                'gender': [gender],
                'length_of_stay': [length_of_stay],
                'num_procedures': [num_procedures],
                'num_medications': [num_medications],
                'num_lab_procedures': [num_lab_procedures],
                'num_diagnoses': [num_diagnoses],
                'blood_pressure_systolic': [systolic_bp],
                'blood_pressure_diastolic': [diastolic_bp],
                'heart_rate': [heart_rate],
                'respiratory_rate': [respiratory_rate],
                'temperature': [temperature],
                'oxygen_saturation': [oxygen_saturation],
                'primary_diagnosis': [primary_diagnosis],
                'emergency_admission': [int(emergency_admission)],
                'insurance': [insurance],
                'marital_status': [marital_status],
                'previous_admissions': [previous_admissions],
                'comorbidity_Diabetes': [int(diabetes)],
                'comorbidity_Hypertension': [int(hypertension)],
                'comorbidity_Heart Disease': [int(heart_disease)],
                'comorbidity_COPD': [int(copd)],
                'comorbidity_Cancer': [int(cancer)],
                'comorbidity_Obesity': [int(obesity)]
            })
            
            try:
                # Preprocess data
                X = preprocessor.transform(patient_data)
                
                # Predict probability
                risk_probability = model.predict_proba(X)[0, 1]
                
                # Generate risk score and label
                risk_score = generate_risk_scores(np.array([risk_probability]), num_bins=num_risk_categories)[0]
                risk_label = generate_risk_labels(np.array([risk_score]), num_bins=num_risk_categories)[0]
                
                # Display results
                st.subheader("Risk Assessment Results")
                
                # Create columns for results
                col1, col2, col3 = st.columns(3)
                
                # Display metrics
                col1.metric("Risk Probability", f"{risk_probability:.1%}")
                col2.metric("Risk Score", risk_score)
                col3.metric("Risk Category", risk_label)
                
                # Display risk gauge
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.barh(y=0, width=100, color='lightgray')
                ax.barh(y=0, width=risk_probability * 100, color='red' if risk_probability >= risk_threshold else 'green')
                ax.set_xlim(0, 100)
                ax.set_ylim(-0.5, 0.5)
                ax.set_xlabel('Risk Probability (%)')
                ax.axvline(x=risk_threshold * 100, color='black', linestyle='--')
                ax.text(risk_threshold * 100 + 2, 0, f'Threshold ({risk_threshold:.0%})', va='center')
                ax.set_yticks([])
                st.pyplot(fig)
                
                # Decision message
                if risk_probability >= risk_threshold:
                    st.error(f"⚠️ HIGH RISK: This patient is at high risk of {target_options[target_variable].lower()}.")
                    st.info("Consider implementing preventive interventions and closer monitoring.")
                else:
                    st.success(f"✅ LOW RISK: This patient is at low risk of {target_options[target_variable].lower()}.")
                
                # Calculate SHAP values for this patient if feature names are available
                if feature_names is not None:
                    st.subheader("Risk Factors for This Patient")
                    explainer = shap.Explainer(model)
                    shap_values = explainer(X)
                    
                    # Create a DataFrame for the SHAP values
                    shap_df = pd.DataFrame({
                        'Feature': feature_names,
                        'SHAP Value': shap_values.values[0],
                        'Absolute Value': np.abs(shap_values.values[0])
                    })
                    
                    # Sort by absolute SHAP value
                    shap_df = shap_df.sort_values('Absolute Value', ascending=False).head(10)
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 5))
                    colors = ['red' if x > 0 else 'blue' for x in shap_df['SHAP Value']]
                    ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
                    ax.set_xlabel('Impact on Risk (SHAP Value)')
                    ax.set_title('Top Factors Influencing Risk Prediction')
                    st.pyplot(fig)
                    
                    # Explanation
                    positive_factors = shap_df[shap_df['SHAP Value'] > 0].sort_values('SHAP Value', ascending=False)
                    negative_factors = shap_df[shap_df['SHAP Value'] < 0].sort_values('SHAP Value')
                    
                    if not positive_factors.empty:
                        st.markdown("**Factors increasing risk:**")
                        for _, row in positive_factors.head(3).iterrows():
                            st.markdown(f"- {row['Feature']}")
                    
                    if not negative_factors.empty:
                        st.markdown("**Factors decreasing risk:**")
                        for _, row in negative_factors.head(3).iterrows():
                            st.markdown(f"- {row['Feature']}")
            
            except Exception as e:
                st.error(f"Error calculating risk: {str(e)}")
                st.info("Please ensure you've entered all required information correctly.")
    
    elif input_method == "Generate Sample Patients":
        st.info("Generate a sample dataset of patients to demonstrate the risk stratification model.")
        
        # Number of patients to generate
        num_patients = st.slider("Number of patients to generate", 10, 100, 20)
        
        if st.button("Generate and Analyze Sample Patients"):
            with st.spinner("Generating sample patients..."):
                # Generate sample data
                sample_df = create_sample_healthcare_dataset(n_samples=num_patients)
                
                # Preprocess data
                X = preprocessor.transform(sample_df.drop(columns=[target_variable]))
                
                # Predict probabilities
                risk_probabilities = model.predict_proba(X)[:, 1]
                
                # Generate risk scores and labels
                risk_scores = generate_risk_scores(risk_probabilities, num_bins=num_risk_categories)
                risk_labels = generate_risk_labels(risk_scores, num_bins=num_risk_categories)
                
                # Add results to dataframe
                results_df = sample_df.copy()
                results_df['Risk Probability'] = risk_probabilities
                results_df['Risk Score'] = risk_scores
                results_df['Risk Category'] = risk_labels
                results_df['High Risk'] = risk_probabilities >= risk_threshold
                
                # Display results
                st.subheader("Sample Patients with Risk Assessment")
                
                # Color high risk patients
                def highlight_high_risk(val):
                    if val == True:
                        return 'background-color: #ffcccc'
                    return ''
                
                # Display styled dataframe
                st.dataframe(
                    results_df.style.applymap(
                        highlight_high_risk, 
                        subset=['High Risk']
                    )
                )
                
                # Plot risk distribution
                st.subheader("Risk Distribution")
                fig = plot_risk_distribution(risk_labels)
                st.pyplot(fig)
                
                # Summary statistics
                st.subheader("Summary Statistics")
                total_patients = len(results_df)
                high_risk_count = results_df['High Risk'].sum()
                high_risk_percentage = high_risk_count / total_patients * 100
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Patients", total_patients)
                col2.metric("High Risk Patients", high_risk_count)
                col3.metric("High Risk Percentage", f"{high_risk_percentage:.1f}%")

# Tab 2: Model Performance
with tab2:
    st.header("Model Performance")
    
    st.info(f"This section shows the performance metrics of the {target_options[target_variable]} prediction model.")
    
    # Load test data if available
    try:
        #st.write("Target variable:", target_variable)
        #st.write("Model path:", model_path)

        #st.write("Current Working Directory:", os.getcwd())
        #st.write("File exists (X_test):", os.path.exists("data/processed/X_test_processed.npy"))
        #st.write("File exists (y_test):", os.path.exists("data/processed/y_test.npy"))

        X_test = np.load("data/processed/X_test_processed.npy")
        y_test = np.load("data/processed/y_test.npy")
        
        # Calculate predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= risk_threshold).astype(int)
        
        # Display metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Display metrics
        st.subheader("Model Performance Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{accuracy:.3f}")
        col2.metric("Precision", f"{precision:.3f}")
        col3.metric("Recall", f"{recall:.3f}")
        col4.metric("F1 Score", f"{f1:.3f}")
        col5.metric("ROC AUC", f"{roc_auc:.3f}")
        
        # Plot ROC curve
        st.subheader("ROC Curve")
        
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        st.pyplot(fig)
        
        # Plot confusion matrix
        st.subheader("Confusion Matrix")
        fig = plot_confusion_matrix(y_test, y_pred)
        st.pyplot(fig)
        
        # Find optimal threshold
        st.subheader("Threshold Optimization")
        
        criteria = st.selectbox(
            "Optimization Criterion",
            options=['balanced', 'f1', 'precision', 'recall'],
            index=0
        )
        
        optimal_threshold = find_optimal_threshold(y_test, y_pred_proba, criterion=criteria)
        st.info(f"Optimal threshold for {criteria} criterion: {optimal_threshold:.3f}")
        
        # Recalculate metrics with optimal threshold
        if st.button("Apply Optimal Threshold"):
            y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
            
            accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
            precision_optimal = precision_score(y_test, y_pred_optimal)
            recall_optimal = recall_score(y_test, y_pred_optimal)
            f1_optimal = f1_score(y_test, y_pred_optimal)
            
            st.subheader("Metrics with Optimal Threshold")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy_optimal:.3f}", f"{accuracy_optimal - accuracy:.3f}")
            col2.metric("Precision", f"{precision_optimal:.3f}", f"{precision_optimal - precision:.3f}")
            col3.metric("Recall", f"{recall_optimal:.3f}", f"{recall_optimal - recall:.3f}")
            col4.metric("F1 Score", f"{f1_optimal:.3f}", f"{f1_optimal - f1:.3f}")
            
            # Updated confusion matrix
            st.subheader("Updated Confusion Matrix")
            fig = plot_confusion_matrix(y_test, y_pred_optimal)
            st.pyplot(fig)
    
    #except Exception as e:
    #    st.warning("Test data not available. Performance metrics cannot be calculated.")
    #    st.info("To view model performance, please ensure you have processed data available in the data/processed directory.")
    except Exception as e:
        st.error(f"⚠️ Exception occurred while loading test data: {str(e)}")


# Tab 3: Model Explanation
with tab3:
    st.header("Model Explanation")
    
    st.info("This section provides insights into the factors that influence the model's predictions.")

    st.subheader("Feature Importance")

    try:
        # Safely convert to list if Series
        if isinstance(feature_names, pd.Series):
            feature_names = feature_names.tolist()
        elif isinstance(feature_names, np.ndarray):
            feature_names = feature_names.tolist()

        # Validate feature names
        #st.write("✅ Feature Names Loaded")
        #st.write("Length:", len(feature_names))
        #st.write("Sample:", feature_names[:5])

        # For tree-based models
        if hasattr(model, 'feature_importances_'):
            # Convert importances to array to avoid Series ambiguity
            importances = np.array(model.feature_importances_)

            st.write("🔍 Feature Names Length:", len(feature_names))
            st.write("🔍 Importances Length:", len(importances))
            st.write("🔍 Sample Feature Names:", feature_names[:5])
            st.write("🔍 Sample Importances:", importances[:5])

            if len(importances) != len(feature_names):
                st.error("❌ Mismatch in lengths of importances and feature_names.")
            else:
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values(by='importance', ascending=False)

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(x='importance', y='feature', data=feature_importance.head(max_features_display), ax=ax)
                ax.set_title("Feature Importance")
                st.pyplot(fig)

                st.dataframe(feature_importance)


        # For linear models
        elif hasattr(model, 'coef_'):
            coefs = model.coef_[0]
            if len(coefs) != len(feature_names):
                st.error("❌ Mismatch in lengths of coefficients and feature_names.")
            else:
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient': coefs,
                    'abs': np.abs(coefs)
                }).sort_values(by='abs', ascending=False)

                feature_importance['color'] = feature_importance['coefficient'].apply(lambda x: 'red' if x > 0 else 'blue')

                fig, ax = plt.subplots(figsize=(10, 8))
                palette = feature_importance.head(max_features_display)['color'].tolist()

                sns.barplot(
                    x='coefficient',
                    y='feature',
                    data=feature_importance.head(max_features_display),
                    palette=palette,
                    ax=ax
                )
                ax.set_title("Feature Coefficients")
                st.pyplot(fig)

                st.dataframe(feature_importance)
        else:
            st.warning("⚠️ Feature importance not available for this model type.")

    except Exception as e:
        st.error(f"🔥 Error: {str(e)}")
        st.code(traceback.format_exc())  # This prints the full traceback


# Tab 4: About
with tab4:
    st.header("About This Project")
    
    st.markdown("""
    ### Patient Risk Stratification Model
    
    This application demonstrates a machine learning model for predicting patient risk of hospital readmission or ICU transfer.
    
    **Key Features:**
    - Risk assessment for individual patients
    - Batch processing for multiple patients
    - Performance evaluation and model explanation
    - Visualization of risk factors
    
    **How It Works:**
    1. **Data Collection**: Patient data is collected from electronic health records
    2. **Preprocessing**: Data is cleaned and transformed for model input
    3. **Risk Prediction**: Machine learning model calculates risk probability
    4. **Stratification**: Patients are categorized into risk groups
    5. **Interpretation**: Key risk factors are identified for each patient
    
    **Benefits for Healthcare Providers:**
    - Early identification of high-risk patients
    - Resource optimization for preventative care
    - Targeted interventions for patients who need them most
    - Data-driven decision making for clinical care
    
    **Technologies Used:**
    - Python (Pandas, NumPy, Scikit-learn)
    - Machine Learning (XGBoost, Random Forest, Logistic Regression)
    - Data Visualization (Matplotlib, Seaborn)
    - Web Application (Streamlit)
    - Model Interpretation (SHAP)
    
    **Developer Information:**
    This project was developed as a portfolio project for a data science role in healthcare.
    """)
    
    # Add GitHub link
    st.info("For more information and source code, visit the [GitHub repository](https://github.com/your-username/patient-risk-stratification)")
    
    # Disclaimer
    st.warning("""
    **Disclaimer:** This application is for demonstration purposes only and should not be used for actual clinical decision making.
    Always consult with qualified healthcare professionals for medical decisions.
    """)

# Main footer
st.markdown("---")
st.markdown("© 2023 Patient Risk Stratification Tool | For demonstration purposes only")