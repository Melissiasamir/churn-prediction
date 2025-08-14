import streamlit as st
import pandas as pd
import joblib
import os

# ====== Load Models and Processing ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Model")

print(f"Model directory: {MODEL_DIR}")
print(f"Directory exists: {os.path.exists(MODEL_DIR)}")

if os.path.exists(MODEL_DIR):
    files = os.listdir(MODEL_DIR)
    print(f"Files in Model directory: {files}")
    
    # Check specific files
    processing_file = os.path.join(MODEL_DIR, "telco_churn_processing.pkl")
    print(f"Processing file exists: {os.path.exists(processing_file)}")

try:
    processing = joblib.load(os.path.join(MODEL_DIR, "telco_churn_processing.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "telco_churn_scaler.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "telco_churn_voting_model.pkl"))
    print("All models loaded successfully!")
    
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()
    
except AttributeError as e:
    st.error(f"Version compatibility issue: {e}")
    st.error("Please retrain your models with the current scikit-learn version")
    st.stop()
    
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


st.set_page_config(page_title="Telco Churn Prediction", page_icon="📊", layout="wide")

st.title("📞 Telco Customer Churn Prediction App")

# ====== Tabs ======
tab1, tab2 = st.tabs(["📥 Data Input", "📊 Prediction Result"])

# ====== Input Form ======
with tab1:
    st.subheader("Enter Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    with col2:
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    col3, col4 = st.columns(2)
    with col3:
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    
    with col4:
        TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1400.0)

    if st.button("✅ Predict Churn"):
        input_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
        }])
        
        # Process the data
        processed_data = processing.transform(input_data)
        scaled_data = scaler.transform(processed_data)
        
        # Prediction
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]  # Churn probability
        
        # Save results to session state
        st.session_state["prediction"] = prediction
        st.session_state["probability"] = probability
        st.success("Prediction complete! Check the 'Prediction Result' tab.")

# ====== Result Tab ======
with tab2:
    st.subheader("Prediction Result")
    if "prediction" in st.session_state:
        pred = st.session_state["prediction"]
        prob = st.session_state["probability"]
        
        if pred == 1:
            st.error(f"⚠ Customer is likely to Churn ({prob*100:.2f}%)")
        else:
            st.success(f"✅ Customer is likely to Stay ({(1-prob)*100:.2f}%)")
    else:
        st.info("Please enter data in the 'Data Input' tab and click Predict.")
