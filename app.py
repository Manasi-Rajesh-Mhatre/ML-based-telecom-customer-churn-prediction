from flask import Flask, render_template, request
from joblib import load
import pandas as pd
import numpy as np
import traceback

app = Flask(__name__)

try:
    # Load all necessary models
    model = load('models/xgboost_classifier.joblib')
    label_encoders = load('models/label_encoders.joblib')
    scaler = load('models/minmax_scaler.joblib')
    cluster_model = load('models/cluster_model.joblib')  # Load your clustering model
    
    # Cluster descriptions (using your provided mapping)
    cluster_reason_map = {
        0: "This customer has been with us for a long time and pays a higher monthly fee. They use important services like online security and backup, and they rarely churn. However, they might need appreciation or loyalty rewards to maintain their trust.",
        1: "This customer is at very high risk of churn. They are new, pay a very high monthly fee, and might feel overwhelmed or underserved. Immediate attention, personalized support, or a better pricing plan may help retain them.",
        2: "This customer is very new and pays a low monthly fee. They might still be exploring our services or trying out basic plans. Engaging them early with onboarding support and showcasing more features could prevent churn.",
        3: "This customer has moderate tenure and uses only essential services. They don't have technical support or security features, which might reduce their perceived value. Offering them more relevant services or a bundle plan could improve retention.",
        4: "This customer has low to mid-level engagement and prefers paperless billing. They aren't strongly loyal yet, but not highly dissatisfied either. Sending personalized offers or incentives now could positively influence their decision to stay."
    }
    
    # Recommendations for each cluster
    cluster_recommendations = {
        0: "Consider offering a loyalty discount or exclusive feature for long-term customers.",
        1: "Immediate outreach with a personalized offer or dedicated account manager.",
        2: "Send onboarding tips and highlight valuable but underused features.",
        3: "Recommend adding security or support services with a bundle discount.",
        4: "Send a limited-time offer or incentive to encourage continued service."
    }
    
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

        # Make prediction
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        confidence = f"{proba * 100:.1f}%"

        # Only determine cluster if customer is predicted to churn
        cluster_info = None
        if prediction == 1:
            # Make a copy for clustering that only includes the features used during training
            # This might need adjustment based on exactly what features were used for clustering
            cluster_features = [
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                'Contract', 'PaperlessBilling', 'PaymentMethod',
                'MonthlyCharges', 'TotalCharges'
            ]
            
            # Only include features that were used in the clustering model
            cluster_input = input_df[cluster_features].copy()
            
            # Get the cluster prediction
            cluster = cluster_model.predict(cluster_input)[0]
            
            # Get the reason and recommendation for this cluster
            cluster_reason = cluster_reason_map.get(cluster, 
                "We've identified this customer as at risk of churn, but the specific pattern is less common.")
            recommendation = cluster_recommendations.get(cluster, 
                "Consider reaching out with a personalized retention offer.")
                
            cluster_info = {
                'cluster': cluster + 1,  # Show as Cluster 1-5 instead of 0-4 for user-friendliness
                'reason': cluster_reason,
                'recommendation': recommendation
            }

        return render_template('index.html',
                               prediction=prediction,
                               confidence=confidence,
                               cluster_info=cluster_info)

    except Exception as e:
        traceback.print_exc()
        return render_template('index.html',
                               error=str(e))

if __name__ == '__main__':
    app.run(debug=True)