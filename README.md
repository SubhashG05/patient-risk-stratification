# Patient Risk Stratification Model

A machine learning-based clinical decision support tool that predicts patient readmission risk and ICU transfer risk using electronic health record data. This application helps healthcare providers identify high-risk patients, prioritize care, and optimize resource allocation.

![Application Screenshot](streamlitapp.png)

## Features

- **Risk Prediction**: Predict patient readmission risk or ICU transfer risk
- **Risk Stratification**: Categorize patients into risk levels for targeted interventions
- **Interactive Dashboard**: User-friendly interface for clinical decision support
- **Model Explainability**: Interpret model predictions to understand risk factors
- **Batch Processing**: Analyze multiple patients simultaneously
- **Performance Metrics**: Visualize model accuracy and other key metrics

## Application Demo

The application is deployed at: [Streamlit Cloud_patient-risk-stratification-model](https://patient-risk-stratification-model.streamlit.app/)

## Installation (Local)

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

## Usage (Local)

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Access the application in your web browser at `http://localhost:8501`.

## Deployment on AWS EC2 with Docker

This section describes how to deploy the app on an AWS EC2 instance using Docker for a production-ready setup.

### Steps:

1. **Provision EC2 Instance**  
   - Launch an EC2 instance (Ubuntu 20.04 recommended).
   - Select appropriate instance type (e.g., t2.micro for testing, t3.medium for production).
   - Configure security group to allow inbound traffic on port 8501 (Streamlit).

2. **Upload Project Files**  
   - From your local machine, use `scp` to upload your zipped project folder:
     ```bash
     scp -i "path/to/your-key.pem" patient_model.zip ubuntu@<EC2-Public-IP>:/home/ubuntu/
     ```
   - On the EC2 instance, unzip the project folder:
     ```bash
     unzip patient_model.zip
     cd "Patient risk stratification model"
     ```

3. **Build Docker Image**  
   - Ensure Docker is installed on the EC2 instance.
   - Build the Docker image:
     ```bash
     sudo docker build -t patient-risk-app .
     ```

4. **Run Docker Container**  
   - Run the container:
     ```bash
     sudo docker run -d -p 8501:8501 patient-risk-app
     ```
   - Access the app at:  
     `http://<EC2-Public-IP>:8501`

5. **Snapshot**  
   - ![EC2 Docker Deployment Screenshot](ec2_docker_deployment.png)

### Notes:
- Consider setting up a custom domain and HTTPS for production.
- For larger data or high traffic, upgrade the EC2 instance type accordingly.
- Always monitor logs for issues:
  ```bash
  sudo docker logs <container-id>
