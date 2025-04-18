import streamlit as st
import joblib
import pandas as pd

# Load models
model = joblib.load('churn_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

st.title("📞 Customer Churn Predictor")

# Input form with EXACT feature names
with st.form("inputs"):
    col1, col2 = st.columns(2)
    
    with col1:
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Senior_Citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        Partner = st.selectbox("Partner", ["No", "Yes"])
        Dependents = st.selectbox("Dependents", ["No", "Yes"])
        Tenure_Months = st.slider("Tenure Months", 1, 72, 12)
        Phone_Service = st.selectbox("Phone Service", ["No", "Yes"])
        Multiple_Lines = st.selectbox("Multiple Lines", ["No", "Yes"])
        
    with col2:
        Internet_Service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        Online_Security = st.selectbox("Online Security", ["No", "Yes"])
        Online_Backup = st.selectbox("Online Backup", ["No", "Yes"])
        Device_Protection = st.selectbox("Device Protection", ["No", "Yes"])
        Tech_Support = st.selectbox("Tech Support", ["No", "Yes"])
        Streaming_TV = st.selectbox("Streaming TV", ["No", "Yes"])
        Streaming_Movies = st.selectbox("Streaming Movies", ["No", "Yes"])
    
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    Paperless_Billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    Payment_Method = st.selectbox("Payment Method", [
        "Electronic check", 
        "Mailed check", 
        "Bank transfer (automatic)", 
        "Credit card (automatic)"
    ])
    Monthly_Charges = st.number_input("Monthly Charges", 0, 200, 65)
    Total_Charges = st.number_input("Total Charges", 0, 10000, 2000)
    
    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare input with EXACT feature names
    input_data = {
        'Gender': Gender,
        'Senior Citizen': Senior_Citizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'Tenure Months': Tenure_Months,
        'Phone Service': Phone_Service,
        'Multiple Lines': Multiple_Lines,
        'Internet Service': Internet_Service,
        'Online Security': Online_Security,
        'Online Backup': Online_Backup,
        'Device Protection': Device_Protection,
        'Tech Support': Tech_Support,
        'Streaming TV': Streaming_TV,
        'Streaming Movies': Streaming_Movies,
        'Contract': Contract,
        'Paperless Billing': Paperless_Billing,
        'Payment Method': Payment_Method,
        'Monthly Charges': Monthly_Charges,
        'Total Charges': Total_Charges
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
  
    # Apply label encoding safely
    for col in label_encoders:
        if col in input_df.columns:
            encoder = label_encoders[col]
            valid_classes = set(encoder.classes_)
            input_df[col] = input_df[col].apply(lambda x: x if x in valid_classes else list(valid_classes)[0])
            input_df[col] = encoder.transform(input_df[col])

    
    # Scale numerical features
    num_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # Predict
    input_df = input_df[model.feature_names_in_]
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    # Show results
    if prediction == 1:
        st.error(f"🚨 High churn risk: {probability*100:.1f}% probability")
    else:
        st.success(f"✅ Low churn risk: {probability*100:.1f}% probability")
    
    # Explain key factors
    st.write("**Top factors influencing this prediction:**")
    st.write("- Contract type")
    st.write("- Tenure duration")
    st.write("- Internet service type")