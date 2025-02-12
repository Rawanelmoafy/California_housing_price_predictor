import streamlit as st
import pandas as pd
import numpy as np
import joblib

model=joblib.load("random_forest_model.pkl")



st.set_page_config(
        page_title="California House Predictor",
        page_icon="mansion.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )

st.title("___🏡California Housing Price Prediction🏡___")
st.image("a232cdb7-5901-4cd3-93ca-a760650f77ee.webp")
st.write("### 🌟 Welcome to the California Housing Price Prediction App!")
st.markdown("This tool helps you estimate house prices based on key features. Enter the details below and get an instant prediction! 🚀")
st.markdown("___")

col1, col2, col3 = st.columns(3)

st.markdown("### Enter House Details:")
with col1:
    longitude = st.number_input("🌍 Longitude")
    latitude = st.number_input("📍 Latitude")
    housing_median_age = st.number_input("🏠 Housing Median Age")

with col2:
    total_rooms = st.number_input("🛏️ Total Rooms")
    total_bedrooms = st.number_input("🛌 Total Bedrooms")
    population = st.number_input("👨‍👩‍👧 Population")
with col3:
    households = st.number_input("🏡 Households")
    median_income = st.number_input("💰 Median Income")

rooms_per_houshold = total_rooms / households if households > 0 else 0
bedrooms_per_room = total_bedrooms / total_rooms if total_rooms > 0 else 0
populations_per_household = population / households if households > 0 else 0

st.markdown("### Auto-Calculated Features:")
st.write(f"- **Rooms per Household**: {rooms_per_houshold:.2f}")
st.write(f"- **Bedrooms per Room**: {bedrooms_per_room:.4f}")
st.write(f"- **Population per Household**: {populations_per_household:.2f}")

ocean_proximity = st.selectbox(
    "🌊 Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)


features_df = pd.DataFrame([[longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
                             population, households, median_income, bedrooms_per_room,
                             rooms_per_houshold, populations_per_household, ocean_proximity]],
                           columns=["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
                                    "population", "households", "median_income", "bedrooms_per_room",
                                    "rooms_per_houshold", "populations_per_household", "ocean_proximity"])

features_transformed = model.named_steps["preprocessor"].transform(features_df) #->this apply what happend in pipeline

if st.button("💡 Predict House Price"):
    prediction = model.named_steps["regressor"].predict(features_transformed)#-> so we save the model in regressor in pipeline so it call it
    st.metric(label="🏠 Estimated House Price", value=f"${prediction[0]:,.2f}") 