"""
Script to generate a realistic dataset for Grape Downy Mildew prediction.
Features: Rainfall (mm), Humidity (%), Temperature (°C), Crop Stage (1-3)
Target: Disease Risk (Low, Medium, High)

The dataset simulates realistic agricultural conditions where:
- High rainfall + high humidity + moderate temperatures = higher disease risk
- Downy mildew thrives in wet, humid conditions (20-30°C optimal)
- Mid growth stage is most susceptible
"""

import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 600

# Generate realistic feature values
rainfall = np.random.uniform(0, 150, n_samples)       # mm (0-150mm range)
humidity = np.random.uniform(40, 100, n_samples)       # % (40-100% range)
temperature = np.random.uniform(10, 40, n_samples)     # °C (10-40°C range)
crop_stage = np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.45, 0.25])  # Early, Mid, Late

# Assign disease risk based on realistic conditions
risk_labels = []
for i in range(n_samples):
    # Calculate a risk score based on environmental conditions
    risk_score = 0
    
    # Rainfall contribution (more rain = more risk)
    if rainfall[i] > 80:
        risk_score += 3
    elif rainfall[i] > 40:
        risk_score += 2
    elif rainfall[i] > 15:
        risk_score += 1
    
    # Humidity contribution (higher humidity = more risk)
    if humidity[i] > 85:
        risk_score += 3
    elif humidity[i] > 70:
        risk_score += 2
    elif humidity[i] > 55:
        risk_score += 1
    
    # Temperature contribution (20-30°C is optimal for downy mildew)
    if 20 <= temperature[i] <= 30:
        risk_score += 3
    elif 15 <= temperature[i] < 20 or 30 < temperature[i] <= 35:
        risk_score += 1
    
    # Crop stage contribution (mid-stage most susceptible)
    if crop_stage[i] == 2:
        risk_score += 2
    elif crop_stage[i] == 1:
        risk_score += 1
    
    # Add some randomness for realism (~10% chance of flipping)
    if np.random.random() < 0.1:
        risk_score = np.random.randint(0, 8)
    
    # Convert score to risk label (adjusted thresholds for better balance)
    if risk_score <= 4:
        risk_labels.append("Low")
    elif risk_score <= 7:
        risk_labels.append("Medium")
    else:
        risk_labels.append("High")

# Create DataFrame
df = pd.DataFrame({
    "Rainfall_mm": np.round(rainfall, 1),
    "Humidity_pct": np.round(humidity, 1),
    "Temperature_C": np.round(temperature, 1),
    "Crop_Stage": crop_stage,
    "Disease_Risk": risk_labels
})

# Introduce a few missing values for realism (~2%)
missing_indices = np.random.choice(n_samples, size=12, replace=False)
for idx in missing_indices[:4]:
    df.loc[idx, "Rainfall_mm"] = np.nan
for idx in missing_indices[4:8]:
    df.loc[idx, "Humidity_pct"] = np.nan
for idx in missing_indices[8:10]:
    df.loc[idx, "Temperature_C"] = np.nan
for idx in missing_indices[10:]:
    df.loc[idx, "Crop_Stage"] = np.nan

# Verify class distribution
print("Class Distribution:")
print(df["Disease_Risk"].value_counts())
print(f"\nTotal samples: {len(df)}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Save to CSV
df.to_csv("C:/project/dataset.csv", index=False)
print("\nDataset saved to C:/project/dataset.csv")
