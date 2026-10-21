# Diabetes Risk Predictor

A lightweight machine-learning web application built with Streamlit that estimates diabetes risk from selected health measurements.

The application provides an educational risk estimate, probability score, risk category, model information, interactive visualizations, prediction history, and downloadable reports.

> **Important:** This application is intended only for academic, educational, and demonstration purposes. Its predictions are not medical diagnoses and must not replace advice, diagnosis, or treatment from a qualified healthcare professional.

---

## Features

The application currently includes:

- Patient name input
- Automatically generated Patient ID
- Prediction date and time
- Health-parameter input form
- Pregnancy-history input
- Diabetes-risk probability
- Application-defined risk categories:
  - Low
  - Moderate
  - High
- Model accuracy display
- ROC-AUC display
- Interactive risk dashboard
- Prediction probability visualization
- Random Forest feature-importance chart
- Current-session prediction history
- CSV history download
- Downloadable HTML prediction report
- Light futuristic Streamlit interface
- Responsive layout
- Medical disclaimer

The application does not display the full dataset to users. It uses the dataset internally for model training.

---

## Dataset

This project uses the **Pima Indians Diabetes Database**.

Dataset source:

[https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

The dataset contains:

- 768 patient records
- 8 input features
- 1 binary target variable named `Outcome`

The input features are:

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`

The `Outcome` column is used as the target variable:

- `0`: No diabetes recorded in the dataset
- `1`: Diabetes recorded in the dataset

The dataset is a historical benchmark dataset and should not be treated as a current clinical dataset. It contains measurements from a specific population and should not be assumed to represent every patient or population.

---

## Machine Learning Model

The application uses a **Random Forest Classifier** from Scikit-learn.

Random Forest is an ensemble model that combines multiple decision trees to produce a classification result and probability estimate.

### Preprocessing

Before training, the application:

1. Loads `diabetes.csv`.
2. Separates the input features from `Outcome`.
3. Treats zero values in selected medical measurements as missing values.
4. Applies median imputation through a Scikit-learn pipeline.
5. Splits the data into training and testing sets.
6. Uses stratification to preserve the class distribution.
7. Trains the model on 80% of the dataset.
8. Evaluates the model on the remaining 20%.

The measurements for which zero values are treated as missing are:

- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`

Pregnancy count and age are not treated as missing when their value is zero.

### Current model configuration

- Algorithm: Random Forest Classifier
- Number of trees: `200`
- Maximum tree depth: `8`
- Class weighting: `balanced`
- Random state: `42`
- Training data: `80%`
- Testing data: `20%`

The model and loaded dataset are cached during the application session to avoid unnecessary repeated work.

---

## Model Performance

Using the current dataset split, preprocessing pipeline, and model configuration, the application reports approximately:

- Accuracy: `72.73%`
- ROC-AUC: `0.83`

These values describe the current implementation only. They may change if the dataset, random split, preprocessing method, or model configuration changes.

The reported performance should not be interpreted as clinical accuracy. It should also not be compared directly with results from other projects or notebooks because different preprocessing methods, feature engineering methods, datasets, splits, and model configurations may have been used.

---

## Risk Categories

The application converts the model probability into an application-defined category:

- **Low:** probability below `0.33`
- **Moderate:** probability from `0.33` to below `0.66`
- **High:** probability of `0.66` or above

These thresholds are visual application thresholds only. They are not clinically validated risk categories and must not be used for medical decision-making.

The displayed probability is the model's estimated probability for the entered values. It is not the same as a confirmed diagnosis.

---

## Application Workflow

The application works as follows:

1. The user enters a patient name.
2. The application displays an automatically generated Patient ID.
3. The user enters the available health measurements.
4. The user submits the prediction form.
5. The application prepares the input values.
6. The model applies the same preprocessing pipeline used during training.
7. The Random Forest model generates a classification and probability.
8. The probability is assigned to a Low, Moderate, or High application category.
9. The result, timestamp, probability, and model metrics are displayed.
10. The prediction is stored in the current Streamlit session.
11. The user can inspect the dashboard and prediction history.
12. The user can download an individual HTML report or current-session history as CSV.

Streamlit forms batch widget inputs until the submit button is selected, helping prevent the model from running every time a slider is adjusted [web:70].

---

## Project Structure

```text
Diabetes-Prediction-Streamlit/
│
├── app.py
├── diabetes.csv
├── diabetes_mod.csv
├── diabetes_classification.ipynb
├── requirements.txt
├── README.md
│
└── .streamlit/
    └── config.toml
```

### File descriptions

| File | Description |
|---|---|
| `app.py` | Main Streamlit application |
| `diabetes.csv` | Dataset used by the current application |
| `diabetes_mod.csv` | Additional feature-engineered dataset from the original project; not required by the current application |
| `diabetes_classification.ipynb` | Original or exploratory notebook |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Optional light theme configuration |
| `README.md` | Project documentation |

The application requires `diabetes.csv` to be in the same project directory as `app.py`.

---

## Technology Stack

- Python 3.12
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- SciPy
- Joblib

Streamlit applications can be started with the `streamlit run` command or its Python module equivalent [web:112][web:113].

---

## Installation

### Requirements

Make sure Python 3.12 or a compatible Python version is installed.

Open PowerShell inside the project folder.

### Create a virtual environment

```powershell
python -m venv .venv
```

### Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `requirements.txt` is not available, install the main packages manually:

```powershell
.\.venv\Scripts\python.exe -m pip install `
    streamlit `
    pandas `
    numpy `
    scikit-learn `
    plotly `
    scipy `
    joblib
```

---

## Optional Theme Configuration

The application is designed with a bright interface rather than a dark theme.

Create the following file:

```text
.streamlit/config.toml
```

Add:

```toml
[theme]
base = "light"
primaryColor = "#1976D2"
backgroundColor = "#F7FBFF"
secondaryBackgroundColor = "#EEF6FC"
textColor = "#17324D"
font = "sans serif"

[client]
showErrorDetails = "none"

[server]
enableXsrfProtection = true
```

The custom styling in `app.py` adds the light gradient background, glass-style cards, blue and cyan controls, soft violet highlights, and lightweight hover animations.

---

## Run the Application

Start the application from the project folder:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Alternatively, if the virtual environment is activated:

```powershell
streamlit run app.py
```

Streamlit normally opens the application automatically in a browser. If it does not, open:

```text
http://localhost:8501
```

Streamlit documents `streamlit run app.py` as the standard way to launch a local application [web:112][web:113].

---

## Dashboard

The dashboard provides information from the current application session, including:

- Total number of predictions
- Model accuracy
- ROC-AUC
- Risk-category distribution
- Prediction probability over time
- Recent prediction results
- Random Forest feature importance

The prediction history is stored in Streamlit session state. It is temporary and is cleared when the session is cleared or the application is restarted.

---

## Reports and Downloads

The application supports two types of downloads.

### Individual prediction report

The user can download an HTML report containing:

- Patient name
- Patient ID
- Timestamp
- Entered health values
- Estimated probability
- Risk category
- Model accuracy
- ROC-AUC
- Medical disclaimer

### Current-session history

The user can download the predictions made during the current session as a CSV file.

The download functionality uses Streamlit's built-in download component [web:15].

---

## Limitations

This project has important limitations:

- The dataset contains only 768 records.
- The dataset represents a specific historical population.
- The dataset is not a current clinical database.
- The model is intended for education and demonstration.
- Predictions may be incorrect.
- Model probability estimates may not be calibrated for clinical use.
- The risk thresholds are application-defined.
- The application does not perform clinical testing.
- The application does not consider every factor related to diabetes.
- The application has not been clinically validated.
- The application should not be used for diagnosis, treatment, screening, or medical decisions.
- Current-session history is not a permanent medical record.
- The application does not provide secure patient-record storage.
- The model should not be deployed in a healthcare setting without extensive validation, security review, monitoring, and regulatory assessment.

---

## Privacy and Security

Do not enter real identifiable patient information into an unsecured development or demonstration application.

The current application:

- Stores prediction history only during the active Streamlit session.
- Does not use a production database.
- Does not provide user authentication.
- Does not provide role-based access control.
- Does not encrypt stored patient data.
- Does not create a permanent medical record.

For real-world use, the application would require appropriate privacy, security, consent, data-retention, and regulatory controls.

---

## Future Improvements

Possible future improvements include:

- External validation using additional datasets
- Comparison of multiple machine-learning algorithms
- Cross-validation
- Calibration analysis
- Confusion matrix
- Precision, recall, and F1-score
- ROC curve and precision-recall curve
- Explainable AI using SHAP
- Better input validation and clinical range checks
- Model versioning
- Model retraining workflow
- Dataset and model monitoring
- Batch prediction from CSV files
- PDF reports
- Secure database storage
- Patient search and history
- User authentication
- Role-based access
- Audit logs
- Cloud deployment
- Accessibility improvements
- Automated tests
- CI/CD integration
- Clinical and regulatory review

---

## Credits

The project uses the Pima Indians Diabetes Database as a machine-learning benchmark dataset.

Dataset source:

[https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

The original project workflow also referenced the following Kaggle notebook:

[https://www.kaggle.com/code/vincentlugat/pima-indians-diabetes-eda-prediction-0-906](https://www.kaggle.com/code/vincentlugat/pima-indians-diabetes-eda-prediction-0-906)

---

## Disclaimer

This project is created for academic, educational, and demonstration purposes only.

The predictions generated by this application are machine-learning estimates. They are not medical diagnoses, clinical assessments, or treatment recommendations.

Do not use this application to make decisions about medication, treatment, emergency care, diagnosis, or other medical matters.

If you have concerns about diabetes or your health, consult a qualified healthcare professional.