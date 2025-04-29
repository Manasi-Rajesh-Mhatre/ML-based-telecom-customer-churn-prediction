from flask import Flask, render_template, request
from joblib import load
import pandas as pd
import traceback

app = Flask(__name__)

try:
    model = load('models/xgboost_classifier.joblib')
    label_encoders = load('models/label_encoders.joblib')
    scaler = load('models/minmax_scaler.joblib')
    
    print("Label Encoder Classes:")
    for feature, le in label_encoders.items():
        print(f"{feature}: {le.classes_}")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    traceback.print_exc()
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form.to_dict()

        value_mapping = {
            'Partner': {'no': 'No', 'yes': 'Yes'},
            'Dependents': {'no': 'No', 'yes': 'Yes'},
            'Contract': {
                'month-to-month': 'Month-to-month',
                'one year': 'One year',
                'two year': 'Two year'
            },
            'PaperlessBilling': {'no': 'No', 'yes': 'Yes'},
            'PaymentMethod': {
                'electronic check': 'Electronic check',
                'mailed check': 'Mailed check',
                'bank transfer (automatic)': 'Bank transfer (automatic)',
                'credit card (automatic)': 'Credit card (automatic)'
            },
            'OnlineSecurity': {'no': 'No', 'yes': 'Yes'},
            'OnlineBackup': {'no': 'No', 'yes': 'Yes'},
            'DeviceProtection': {'no': 'No', 'yes': 'Yes'},
            'TechSupport': {'no': 'No', 'yes': 'Yes'}
        }

        input_data = {
            'SeniorCitizen': int(form_data['SeniorCitizen']),
            'Partner': value_mapping['Partner'][form_data['Partner']],
            'Dependents': value_mapping['Dependents'][form_data['Dependents']],
            'tenure': float(form_data['tenure']),
            'Contract': value_mapping['Contract'][form_data['Contract']],
            'PaperlessBilling': value_mapping['PaperlessBilling'][form_data['PaperlessBilling']],
            'PaymentMethod': value_mapping['PaymentMethod'][form_data['PaymentMethod']],
            'MonthlyCharges': float(form_data['MonthlyCharges']),
            'TotalCharges': float(form_data['TotalCharges']),
            'OnlineSecurity': value_mapping['OnlineSecurity'][form_data['OnlineSecurity']],
            'OnlineBackup': value_mapping['OnlineBackup'][form_data['OnlineBackup']],
            'DeviceProtection': value_mapping['DeviceProtection'][form_data['DeviceProtection']],
            'TechSupport': value_mapping['TechSupport'][form_data['TechSupport']],
        }

        input_df = pd.DataFrame([input_data])

        categorical_features = [
            'Partner', 'Dependents', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]

        for feature in categorical_features:
            if feature in label_encoders:
                input_df[feature] = label_encoders[feature].transform(input_df[feature])

        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])

        expected_order = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                          'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                          'Contract', 'PaperlessBilling', 'PaymentMethod',
                          'MonthlyCharges', 'TotalCharges']
        input_df = input_df[expected_order]

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        confidence = f"{proba * 100:.1f}%"

        return render_template('index.html',
                               prediction=prediction,
                               confidence=confidence)

    except Exception as e:
        traceback.print_exc()
        return render_template('index.html',
                               error=str(e))

if __name__ == '__main__':
    app.run(debug=True)