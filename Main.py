import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load CSV file
file_path = input("Enter CSV file path: ")
df = pd.read_csv(file_path)

print("\nColumns in dataset:")
print(df.columns.tolist())

# Select target column
target_column = input("\nEnter target column name: ")

# Convert target to numeric
df[target_column] = pd.to_numeric(df[target_column], errors="coerce")

# Keep only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove target from features
if target_column in numeric_cols:
    numeric_cols.remove(target_column)

# Fill missing values
df = df.fillna(df.mean(numeric_only=True))

# Features and target
X = df[numeric_cols]
y = df[target_column]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
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

print(f"\nMAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:100], label="Actual")
plt.plot(predictions[:100], label="Predicted")
plt.legend()
plt.title("Prediction Results")
plt.show()

# Predict from a new row in CSV
sample = X.iloc[-1:].copy()
prediction = model.predict(sample)

print("\nPrediction for last row:")
print(prediction[0])