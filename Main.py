import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =========================
# LOAD DATASET
# =========================
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

# Convert target column to numeric
df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"],
    errors="coerce"
)

# Remove missing values
df = df.dropna(subset=["Global_active_power"])

# Use first 50,000 rows for faster training
df = df.iloc[:50000]

# =========================
# FEATURE ENGINEERING
# =========================
df["Hour"] = df["Datetime"].dt.hour
df["Day"] = df["Datetime"].dt.day
df["Month"] = df["Datetime"].dt.month
df["Weekday"] = df["Datetime"].dt.weekday

# Lag Features
df["Lag1"] = df["Global_active_power"].shift(1)
df["Lag2"] = df["Global_active_power"].shift(2)
df["Lag3"] = df["Global_active_power"].shift(3)

df = df.dropna()

# Features and Target
X = df[
    [
        "Hour",
        "Day",
        "Month",
        "Weekday",
        "Lag1",
        "Lag2",
        "Lag3"
    ]
]

y = df["Global_active_power"]

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# =========================
# TRAIN MODEL
# =========================
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# EVALUATE MODEL
# =========================
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\nMAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")

# =========================
# PLOT TEST RESULTS
# =========================
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:200], label="Actual")
plt.plot(predictions[:200], label="Predicted")
plt.legend()
plt.title("Electricity Usage Prediction")
plt.show()

# =========================
# PREDICT NEXT VALUE
# =========================
latest = X.iloc[-1:].copy()
next_usage = model.predict(latest)

print("\nPredicted Next Power Usage:")
print(f"{next_usage[0]:.4f} kW")

# =========================
# FORECAST NEXT 5 DAYS
# =========================

hours_to_predict = 24 * 5  # 5 days

future_predictions = []

last_datetime = df["Datetime"].iloc[-1]

lag1 = df["Global_active_power"].iloc[-1]
lag2 = df["Global_active_power"].iloc[-2]
lag3 = df["Global_active_power"].iloc[-3]

future_dates = []

for i in range(hours_to_predict):

    future_time = last_datetime + pd.Timedelta(hours=i + 1)

    future_row = pd.DataFrame({
        "Hour": [future_time.hour],
        "Day": [future_time.day],
        "Month": [future_time.month],
        "Weekday": [future_time.weekday()],
        "Lag1": [lag1],
        "Lag2": [lag2],
        "Lag3": [lag3]
    })

    pred = model.predict(future_row)[0]

    future_predictions.append(pred)
    future_dates.append(future_time)

    # Update lags for recursive forecasting
    lag3 = lag2
    lag2 = lag1
    lag1 = pred

# =========================
# SAVE FORECAST
# =========================
forecast_df = pd.DataFrame({
    "Datetime": future_dates,
    "Predicted_Global_Active_Power": future_predictions
})

forecast_df.to_csv(
    "next_5_days_power_forecast.csv",
    index=False
)

print("\nForecast saved to:")
print("next_5_days_power_forecast.csv")

print("\nFirst 10 Future Predictions:")
print(forecast_df.head(10))

# =========================
# DAILY TOTAL POWER
# =========================
daily_forecast = (
    forecast_df
    .set_index("Datetime")
    .resample("D")
    .sum()
)

print("\nPredicted Power Consumption For Next 5 Days:")
print(daily_forecast)

# =========================
# PLOT 5-DAY FORECAST
# =========================
plt.figure(figsize=(14, 6))
plt.plot(
    forecast_df["Datetime"],
    forecast_df["Predicted_Global_Active_Power"]
)
plt.title("Predicted Electricity Usage - Next 5 Days")
plt.xlabel("Datetime")
plt.ylabel("Global Active Power")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()