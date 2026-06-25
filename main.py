import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def generate_synthetic_data(num_records=10500):
    """Generates 10,000+ realistic customer records for the model"""
    np.random.seed(42)
    data = {
        'tenure_months': np.random.randint(1, 72, num_records),
        'monthly_charges': np.random.uniform(20.0, 120.0, num_records),
        'total_charges': np.random.uniform(20.0, 8000.0, num_records),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], num_records),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], num_records),
        'churn': np.random.choice(['Yes', 'No'], num_records, p=[0.25, 0.75]) 
    }
    df = pd.DataFrame(data)
    
    # Intentionally adding missing values to demonstrate data cleaning skills
    df.loc[np.random.choice(df.index, 150), 'total_charges'] = np.nan
    return df

def main():
    print("--- 1. Data Engineering & Preprocessing ---")
    df = generate_synthetic_data()
    print(f"Dataset generated with {len(df)} records.")
    
    # Handling missing values (Resume Bullet 1)
    df['total_charges'] = df['total_charges'].fillna(df['total_charges'].median())
    
    # Encoding categorical data (Resume Bullet 1)
    le = LabelEncoder()
    df['churn'] = le.fit_transform(df['churn']) # Yes=1, No=0
    df = pd.get_dummies(df, columns=['contract_type', 'internet_service'], drop_first=True)

    # Splitting Features (X) and Target (y)
    X = df.drop('churn', axis=1)
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling Numerical Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- 2. Model Training & Evaluation ---")
    # Logistic Regression Model (Resume Bullet 2)
    lr_model = LogisticRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_preds) * 100:.2f}%")

    # Random Forest Model (Resume Bullet 2)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_preds = rf_model.predict(X_test_scaled)
    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds) * 100:.2f}%")

    print("\n--- 3. MLOps Pipeline & Deployment Prep ---")
    # Saving the model and scaler using Joblib (Resume Bullet 3)
    joblib.dump(rf_model, 'rf_churn_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Pipeline established: 'rf_churn_model.pkl' and 'scaler.pkl' saved successfully for deployment.")

if __name__ == "__main__":
    main()
