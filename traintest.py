# Change log:
# 2025-04-16 02:35 PM     Resized visuals
# 2025-04-16 02:40 PM     Modified test size to 20% (To test git branching)

# This is a comment added for branch01

# Predicting House Prices with Linear Regression

# Command to run:
# cd C:\Users\cy185005\portableapps\PortableGit\bin\house-price-prediction
# python c:/Users/cy185005/portableapps/PortableGit/bin/house-price-prediction/traintest.py

import pandas as pd

# pip install scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# pip install matplotlib
# pip install seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("melb_data.csv.zip")
df = df.dropna()

# Select features
features = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = df[features] # Independent variables
y = df['Price'] # Dependent variable

# Linear regression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Model score:", model.score(X_test, y_test))

# Sample visualizations

print("Generating sample visualizations as PNG files...")

plt.figure(figsize=(30,18))
sns.scatterplot(x=df['Rooms'], y=df['Price'])
plt.title("Rooms vs Price")
plt.xlabel("Rooms")
plt.ylabel("Price")
plt.grid(True,"major","both")
plt.savefig('rooms_vs_price.png')

plt.figure(figsize=(30,18))
sns.scatterplot(x=df['BuildingArea'], y=df['Price'])
plt.title("Building Area vs Price")
plt.xlabel("Building Area")
plt.ylabel("Price")
plt.grid(True,"major","both")
plt.savefig('buildingarea_vs_price.png')
