# Patient Risk Stratification Model

A machine learning-based clinical decision support tool that predicts patient readmission risk and ICU transfer risk using electronic health record data. This application helps healthcare providers identify high-risk patients, prioritize care, and optimize resource allocation.

![Application Screenshot](app_screenshot.png)

## Features

- **Risk Prediction**: Predict patient readmission risk or ICU transfer risk
- **Risk Stratification**: Categorize patients into risk levels for targeted interventions
- **Interactive Dashboard**: User-friendly interface for clinical decision support
- **Model Explainability**: Interpret model predictions to understand risk factors
- **Batch Processing**: Analyze multiple patients simultaneously
- **Performance Metrics**: Visualize model accuracy and other key metrics

## Application Demo

The application is deployed at: [Streamlit Cloud](https://patient-risk-stratification.streamlit.app)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/patient-risk-stratification.git
cd patient-risk-stratification
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Access the application in your web browser at `http://localhost:8501`

3. Choose one of the following options:
   - Upload a CSV file with patient data
   - Enter patient information manually
   - Generate sample patients for demonstration

4. View risk assessment results and model explanations

## Project Structure

```
patient_risk_stratification/
├── data/                      # Raw and processed data
├── models/                    # Saved model files
├── src/                       # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py  # Data cleaning and feature engineering
│   ├── model_training.py      # Model development and evaluation
│   └── utils.py               # Helper functions
├── app.py                     # Streamlit application
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

## Model Details

- **Algorithms**: XGBoost, Random Forest, Logistic Regression
- **Features**: Demographic information, vital signs, diagnoses, procedures, medications, lab results
- **Target Variables**: Hospital readmission risk, ICU transfer risk
- **Performance**: Evaluated using accuracy, precision, recall, F1-score, and ROC AUC

## Development

This project was developed as a portfolio project for a data science role in healthcare. The key steps in the development process were:

1. **Data Collection**: Utilizing open-source healthcare datasets for model training
2. **Exploratory Data Analysis**: Understanding the characteristics and patterns in patient data
3. **Feature Engineering**: Creating relevant features for risk prediction
4. **Model Development**: Training and evaluating different machine learning algorithms
5. **Application Development**: Building an interactive web application using Streamlit
6. **Model Interpretation**: Implementing SHAP for model explainability

## Future Improvements

- Implement a time-series analysis component for longitudinal patient data
- Add support for more risk prediction targets (e.g., mortality, length of stay)
- Integrate with external EHR systems for real-time data access
- Develop a clinical validation framework for model verification
- Add collaborative filtering for treatment recommendations

## Contributing

Contributions to this project are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This application is for demonstration purposes only and should not be used for actual clinical decision making. Always consult with qualified healthcare professionals for medical decisions.

## Contact

- GitHub: https://github.com/SubhashG05
- LinkedIn: https://www.linkedin.com/in/g-subhash/
- Email: subhashg5397@gmail.com
