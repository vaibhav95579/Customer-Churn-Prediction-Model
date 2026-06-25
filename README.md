# Customer Churn Prediction Model 📊🤖

## Overview
This project is an end-to-end Machine Learning pipeline designed to predict customer attrition. By analyzing customer data such as tenure, contract type, and monthly charges, the model identifies customers who are at a high risk of canceling their service. 

## Key Features & Workflow
* **Data Engineering:** Preprocessed a dataset of over 10,500 customer records. Handled missing values using median imputation and converted categorical data using One-Hot Encoding and Label Encoding.
* **Predictive Modeling:** Trained and evaluated multiple classification algorithms, primarily comparing **Logistic Regression** and **Random Forest Classifiers**. 
* **Performance:** Achieved a stable ~85% accuracy rate using the Random Forest model.
* **MLOps Integration:** Established a foundational data pipeline. Exported the trained model and data scaler as serialized `.pkl` files using `joblib`, making them ready for API deployment.

## Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Model Serialization:** Joblib

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Execute the pipeline: `python main.py`
3. The script will generate the synthetic dataset, train the models, output the accuracy metrics, and save the model files locally.
