import streamlit as st
import pandas as pd
import joblib

# Load SVM pipeline
model = joblib.load("artifacts/svm_pipeline.pkl")

st.set_page_config(page_title=" Buyer Purchase Decision", layout="wide")
st.title("Real Estate Purchase Decision Prediction")

# User input fields (raw inputs only — pipeline handles encoding!)
country = st.selectbox("Country", ["France", "South Africa", "Germany", "Canada", "Brazil", "UAE", "Australia", "India"])
city = st.text_input("City", "Marseille")
property_type = st.selectbox("Property Type", ["Apartment", "Villa", "Farmhouse", "Townhouse", "Studio"])
furnishing_status = st.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Fully-Furnished"])
property_size_sqft = st.number_input("Property Size (sqft)", 200, 20000, 1000)
price = st.number_input("Price (₹)", 10000, 50000000, 500000)
constructed_year = st.number_input("Construction Year", 1950, 2030, 2000)
previous_owners = st.number_input("Prev Owners", 0, 10, 1)
rooms = st.number_input("Rooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1, 10, 2)
garage = st.number_input("Garage (0/1)", 0, 1, 1)
garden = st.number_input("Garden (0/1)", 0, 1, 0)
crime_cases_reported = st.number_input("Crime Cases", 0, 100, 5)
legal_cases_on_property = st.number_input("Legal Cases (0/1)", 0, 1, 0)
customer_salary = st.number_input("Salary (₹)", 10000, 2000000, 50000)
loan_amount = st.number_input("Loan Amount (₹)", 10000, 20000000, 1000000)
loan_tenure_years = st.number_input("Loan Tenure", 1, 50, 20)
monthly_expenses = st.number_input("Monthly Expenses", 1000, 100000, 10000)
down_payment = st.number_input("Down Payment", 0, 20000000, 500000)
emi_to_income_ratio = monthly_expenses / max(customer_salary, 1)
satisfaction_score = st.slider("Satisfaction", 1, 10, 5)
neighbourhood_rating = st.slider("Neighbourhood Rating", 1, 10, 7)
connectivity_score = st.slider("Connectivity Score", 1, 10, 6)

input_df = pd.DataFrame([locals()])

if st.button("Predict"):
    proba = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.write("Purchase Probability:", f"{proba*100:.2f}%")
    if pred == 1:
        st.success(" Likely to Purchase!")
    else:
        st.error("Not Likely to Purchase.")
