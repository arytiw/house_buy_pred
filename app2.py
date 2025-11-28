import streamlit as st
import pandas as pd
import joblib

# Load SVM pipeline (already includes encoding + scaling + model)
model = joblib.load("svm_pipeline.pkl")

st.set_page_config(page_title="Buyer Purchase Decision", layout="wide")
st.title("🏡 Real Estate Purchase Decision Prediction")

# ----------------------------
# 🔹 User Input Form
# ----------------------------

country = st.selectbox("Country", ["France", "South Africa", "Germany", "Canada", "Brazil", "UAE", "Australia", "India"])
city = st.text_input("City", "Marseille")
property_type = st.selectbox("Property Type", ["Apartment", "Villa", "Farmhouse", "Townhouse", "Studio"])
furnishing_status = st.selectbox("Furnishing Level", ["Unfurnished", "Semi-Furnished", "Fully-Furnished"])
property_size_sqft = st.number_input("Property Size (sqft)", 200, 20000, 1000)
price = st.number_input("Price (₹)", 10000, 50000000, 500000)
constructed_year = st.number_input("Construction Year", 1950, 2030, 2000)
previous_owners = st.number_input("Previous Owners", 0, 10, 1)
rooms = st.number_input("Rooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1, 10, 2)
garage = st.number_input("Garage (0 = No, 1 = Yes)", 0, 1, 1)
garden = st.number_input("Garden (0 = No, 1 = Yes)", 0, 1, 0)
crime_cases_reported = st.number_input("Crime Cases", 0, 500, 5)
legal_cases_on_property = st.number_input("Legal Cases (0 = No, 1 = Yes)", 0, 1, 0)
customer_salary = st.number_input("Customer Salary (₹)", 10000, 2000000, 50000)
loan_amount = st.number_input("Loan Amount (₹)", 10000, 20000000, 1000000)
loan_tenure_years = st.number_input("Loan Tenure (Years)", 1, 50, 20)
monthly_expenses = st.number_input("Monthly Expenses (₹)", 1000, 100000, 10000)
down_payment = st.number_input("Down Payment (₹)", 0, 20000000, 500000)
emi_to_income_ratio = monthly_expenses / max(customer_salary, 1)
satisfaction_score = st.slider("Satisfaction Score", 1, 10, 5)
neighbourhood_rating = st.slider("Neighbourhood Rating", 1, 10, 7)
connectivity_score = st.slider("Connectivity Score", 1, 10, 6)

# Form data into DataFrame
input_df = pd.DataFrame([{
    "country": country,
    "city": city,
    "property_type": property_type,
    "furnishing_status": furnishing_status,
    "property_size_sqft": property_size_sqft,
    "price": price,
    "constructed_year": constructed_year,
    "previous_owners": previous_owners,
    "rooms": rooms,
    "bathrooms": bathrooms,
    "garage": garage,
    "garden": garden,
    "crime_cases_reported": crime_cases_reported,
    "legal_cases_on_property": legal_cases_on_property,
    "customer_salary": customer_salary,
    "loan_amount": loan_amount,
    "loan_tenure_years": loan_tenure_years,
    "monthly_expenses": monthly_expenses,
    "down_payment": down_payment,
    "emi_to_income_ratio": emi_to_income_ratio,
    "satisfaction_score": satisfaction_score,
    "neighbourhood_rating": neighbourhood_rating,
    "connectivity_score": connectivity_score
}])

# ----------------------------
# 🔍 Prediction Button + Override Logic
# ----------------------------
if st.button("Predict"):
    proba = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.subheader("📌 Prediction Result")
    st.metric("Estimated Purchase Probability", f"{proba*100:.2f}%")

    # Financial Rule Override (based on dataset pattern)
    affordable_tenure = loan_tenure_years <= 20
    low_expenses = monthly_expenses <= 10000
    sufficient_down_payment = down_payment >= (price * 0.20)

    if affordable_tenure and low_expenses and sufficient_down_payment:
        st.success("The buyer has a strong financial profile — Likely to Purchase!")
    else:
        if pred == 1:
            st.success("The buyer is LIKELY to purchase this property!")
        else:
            st.error("The buyer is NOT likely to purchase this property.")
