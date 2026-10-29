import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "delivery_data_Tandrima.csv")

# Load the dataset
data = pd.read_csv(DATA_PATH)

print("\nFirst five records:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nDataset information:")
data.info()

print("\nMissing values:")
print(data.isnull().sum())

print("\nStatistical summary:")
print(data.describe())

print("\nCorrelation with delivery time:")
print(data.corr(numeric_only=True)["delivery_time_min"].sort_values(ascending=False))

plt.scatter(data["distance_km"], data["delivery_time_min"])
plt.xlabel("Distance (km)")
plt.ylabel("Delivery time (minutes)")
plt.title("Distance vs Delivery Time")
plt.grid(True)
plt.show()
