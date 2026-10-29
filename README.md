# Food Delivery Time Predictor

Linear Regression model that estimates food delivery time from distance, preparation time, traffic, weather and rider experience.

## Setup
    python -m venv food_delivery_app
    source food_delivery_app/bin/activate      (Windows: food_delivery_app\Scripts\activate)
    pip install -r requirements.txt

## Run order
    python 01_create_data.py      # creates delivery_data_Tandrima.csv
    python 02_explore_data.py     # optional: EDA plot
    python 03_train_model.py      # trains and saves delivery_model.pkl
    python 04_predict.py          # command-line prediction
    streamlit run app.py          # web app

Keep all files in the same folder. Scripts resolve paths relative to their own location.

## Notes
- Training data covers distance 1-15 km, preparation 5-30 min, traffic 1-4, weather 0-3, experience 0-10 years. Predictions outside these ranges are extrapolations.
- delivery_model.pkl depends on the scikit-learn version. If loading fails, re-run 03_train_model.py in your environment.
