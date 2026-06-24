# ⚡ Electricity Usage Predictor

A Machine Learning project that predicts household electricity consumption using historical power usage data and time-based features.

## 📌 Project Overview

This project uses a Random Forest Regressor to predict the next electricity usage value based on:

- Time information (Hour, Day, Month, Weekday)
- Previous power consumption values (Lag Features)

The model is trained on the Household Power Consumption dataset and evaluates its performance using MAE and R² Score.

---

## 🚀 Features

- Data preprocessing and cleaning
- Date-Time feature extraction
- Lag feature generation
- Random Forest Regression model
- Model evaluation with MAE and R² Score
- Visualization of actual vs predicted values
- Next electricity usage prediction

---

## 📂 Dataset

Dataset used:

**Individual Household Electric Power Consumption Dataset**

Source:
https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

Download:
https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib

---

## 📊 Machine Learning Workflow

### 1. Load Dataset

Read the electricity consumption data from the text file.

### 2. Data Cleaning

- Convert power values to numeric format
- Remove missing values

### 3. Feature Engineering

Extract:

- Hour
- Day
- Month
- Weekday

from the DateTime column.

### 4. Create Lag Features

Generate:

- Lag1 (Previous value)
- Lag2 (2 steps back)
- Lag3 (3 steps back)

These help the model learn historical consumption patterns.

### 5. Train-Test Split

- 80% Training Data
- 20% Testing Data

Time order is preserved using:

```python
shuffle=False