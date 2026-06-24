import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(
    "household_power_consumption.txt",
    sep=";",
    low_memory=False,
    na_values=["?"]
)

# Combine Date and Time
df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H:%M:%S"
)

# Select target column
df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"],
    errors="coerce"
)

# Remove missing values
df = df.dropna(subset=["Global_active_power"])

# Use first 50,000 rows for faster training
df = df.iloc[:50000]

# Feature Engineering
df["Hour"] = df["Datetime"].dt.hour
df["Day"] = df["Datetime"].dt.day
df["Month"] = df["Datetime"].dt.month
df["Weekday"] = df["Datetime"].dt.weekday

# Create lag features
df["Lag1"] = df["Global_active_power"].shift(1)
df["Lag2"] = df["Global_active_power"].shift(2)
df["Lag3"] = df["Global_active_power"].shift(3)

df = df.dropna()

# Features and target
X = df[
    ["Hour", "Day", "Month", "Weekday",
     "Lag1", "Lag2", "Lag3"]
]

y = df["Global_active_power"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False
)

# Train Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")

# Plot
plt.figure(figsize=(12,6))
plt.plot(y_test.values[:200], label="Actual")
plt.plot(predictions[:200], label="Predicted")
plt.legend()
plt.title("Electricity Usage Prediction")
plt.show()

# Predict next value
latest = X.iloc[-1:].copy()
next_usage = model.predict(latest)

print("\nPredicted Next Power Usage:")
print(next_usage[0])