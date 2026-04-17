
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="Pammi123/tourism-model", filename="best_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Package Purchase Prediction

st.title("Tourism Package Prediction Application")
st.write("The Tourism Package Prediction App is an internal tool for company policymakers that predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them.")
st.write("Kindly enter the customer details to check whether they are likely to purchase the Wellness Tourism Package or not.")

# Collect user input
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=5, max_value=130, value=15)
Age = st.number_input("Age of the Customer", min_value=18, max_value=65, value=35)
NumberOfPersonVisiting = st.number_input("Number of People accompanying the Customer on the Trip", min_value=1, max_value=6, value=3)
NumberOfFollowups = st.slider("Number of follow-ups by the salesperson after the sales pitch", 1, 6, 4)
NumberOfTrips = st.number_input("Average number of trips the customer takes annually", min_value=1, max_value=22, value=3)
NumberOfChildrenVisiting = st.slider("Number of children below age 5 accompanying the customer", 0, 4, 1)
MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, max_value=99000.0, value=22000.0)
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)

PreferredPropertyStar = st.selectbox("Preferred hotel rating by the customer", [1, 2, 3, 4, 5])
CityTier = st.selectbox("City Tier", [1, 2, 3])

Passport = st.selectbox("Has Passport?", ["Yes", "No"])
OwnCar = st.selectbox("Owns Car?", ["Yes", "No"])

TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])


input_data = pd.DataFrame([{
    'DurationOfPitch': DurationOfPitch,
    'Age': Age,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'CityTier': CityTier,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'MaritalStatus': MaritalStatus,
    'ProductPitched': ProductPitched,
    'Designation': Designation
}])

# Setting the classification threshold
classification_threshold = 0.6

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase package" if prediction == 1 else "NOT purchase package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
