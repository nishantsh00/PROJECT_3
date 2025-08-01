import streamlit as st
import joblib
import numpy as np

model = joblib.load("xgb_smartpremium_model.pkl")


st.title("💡 SmartPremium - Insurance Premium Predictor")

# ----- 1. User Inputs -----
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"])
location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
smoking = st.selectbox("Smoking Status", ["Yes", "No"])
property_type = st.selectbox("Property Type", ["Apartment", "Condo", "House"])

age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2, step=1)
education = st.number_input("Education Level (0 = low, 5 = high)", min_value=0, max_value=5, value=3, step=1)
health = st.number_input("Health Score", min_value=0, max_value=100, value=75, step=1)
policy_type = st.number_input("Policy Type (0 = basic, 2 = premium)", min_value=0, max_value=2, value=1, step=1)
vehicle_age = st.number_input("Vehicle Age (in years)", min_value=0, max_value=30, value=5, step=1)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700, step=1)
insurance_duration = st.number_input("Insurance Duration (years)", min_value=0, max_value=30, value=3, step=1)
feedback = st.number_input("Customer Feedback Score", min_value=0, max_value=10, value=7, step=1)
exercise = st.number_input("Exercise Frequency (0 = none, 3 = high)", min_value=0, max_value=3, value=2, step=1)
income = st.number_input("Annual Income (₹)", min_value=10000, max_value=10000000, step=1000)
prev_claims = st.number_input("Previous Claims (capped)", min_value=0, max_value=50, step=1)

# ----- 2. One-hot Encoding (manual) -----
gender_female = 1 if gender == "Female" else 0
gender_male = 1 if gender == "Male" else 0

marital_divorced = 1 if marital_status == "Divorced" else 0
marital_married = 1 if marital_status == "Married" else 0
marital_single = 1 if marital_status == "Single" else 0

occupation_emp = 1 if occupation == "Employed" else 0
occupation_self = 1 if occupation == "Self-Employed" else 0
occupation_unemp = 1 if occupation == "Unemployed" else 0

loc_rural = 1 if location == "Rural" else 0
loc_suburban = 1 if location == "Suburban" else 0
loc_urban = 1 if location == "Urban" else 0

smoke_yes = 1 if smoking == "Yes" else 0
smoke_no = 1 if smoking == "No" else 0

prop_apartment = 1 if property_type == "Apartment" else 0
prop_condo = 1 if property_type == "Condo" else 0
prop_house = 1 if property_type == "House" else 0

# ----- 3. Create Input Array -----
input_data = np.array([[
    gender_female, gender_male,
    marital_divorced, marital_married, marital_single,
    occupation_emp, occupation_self, occupation_unemp,
    loc_rural, loc_suburban, loc_urban,
    smoke_no, smoke_yes,
    prop_apartment, prop_condo, prop_house,
    age, dependents, education, health,
    policy_type, vehicle_age, credit_score,
    insurance_duration, feedback, exercise,
    income, prev_claims
]])

# ----- 4. Predict -----
if st.button("Predict Premium"):
    log_prediction = model.predict(input_data)[0]
    prediction = np.exp(log_prediction)  # Convert log back to original value
    st.success(f"💰 Predicted Insurance Premium: ₹{prediction:,.2f}")
