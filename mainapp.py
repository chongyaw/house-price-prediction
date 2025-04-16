# Predicting House Prices with Linear Regression

# Command to run:
# cd C:\Users\cy185005\portableapps\PortableGit\bin\house-price-prediction
# streamlit run c:/Users/cy185005/portableapps/PortableGit/bin/house-price-prediction/mainapp.py

import streamlit as st
import pandas as pd

# pip install scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("melb_data.csv.zip")
df = df.dropna()

# Select features
features = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = df[features] # Independent variables
y = df['Price'] # Dependent variable

# Linear regression
model = LinearRegression().fit(X, y)

# Input / Output using Streamlit

st.title("🏠 Melbourne House Price Predictor")

# User input
rooms = st.slider("Number of Rooms", 1, 10, 3)
distance = st.number_input("Distance to CBD (km)", min_value=0.0, max_value=50.0, value=10.0)
landsize = st.number_input("Land Size (m²)", min_value=0, max_value=1000, value=300)
building_area = st.number_input("Building Area (m²)", min_value=0, max_value=1000, value=120)
year_built = st.number_input("Year Built", min_value=1850, max_value=2025, value=1990)

# Prediction
input_data = pd.DataFrame([[rooms, distance, landsize, building_area, year_built]],
                          columns=features)

prediction = model.predict(input_data)[0]

st.subheader(f"💰 Estimated Price: ${prediction:,.0f}")
