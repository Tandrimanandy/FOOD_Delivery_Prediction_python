import os
import pickle
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "delivery_model.pkl")

# STEP 1: Load the saved model
with open(MODEL_PATH, "rb") as file:
    model_package = pickle.load(file)

model = model_package["model"]
features = model_package["features"]
print("Saved model loaded successfully!")


def ask_number(prompt, kind=float, min_value=None, max_value=None):
    """Keep asking until the user enters a valid number inside the range."""
    while True:
        try:
            value = kind(input(prompt))
        except ValueError:
            print("  Please enter a valid number.")
            continue
        if min_value is not None and value < min_value:
            print(f"  Value must be at least {min_value}.")
            continue
        if max_value is not None and value > max_value:
            print(f"  Value must be at most {max_value}.")
            continue
        return value


# STEP 2: Enter details for a new delivery
distance_km = ask_number("Enter distance in km: ", float, min_value=0.1)
preparation_time_min = ask_number("Enter food preparation time in minutes: ", float, min_value=1)
traffic_level = ask_number("Enter traffic level (1-4): ", int, 1, 4)
weather_severity = ask_number("Enter weather severity (0-3): ", int, 0, 3)
rider_experience_years = ask_number("Enter rider experience in years: ", float, min_value=0)

# STEP 3: Prepare input in the correct column order
new_delivery = pd.DataFrame(
    [[distance_km, preparation_time_min, traffic_level,
      weather_severity, rider_experience_years]],
    columns=features,
)

# STEP 4: Predict
predicted_time = model.predict(new_delivery)[0]

# STEP 5: Display
print("\nDELIVERY PREDICTION")
print("-------------------")
print("Estimated delivery time:", round(max(0, predicted_time), 1), "minutes")
