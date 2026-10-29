import os
import pickle

import pandas as pd
import streamlit as st

# STEP 1: Configure the web application
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    layout="centered",
)

# STEP 2: Custom styling (brown and golden yellow theme)
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #2B1D16 0%, #4E342E 100%);
        color: #FFF8E1;
    }
    .hero {
        text-align: center;
        padding: 1.5rem 1rem 1.2rem 1rem;
        border-bottom: 2px solid #C9A227;
        margin-bottom: 1.5rem;
    }
    .hero-icon {
        width: 92px;
        height: 92px;
        border-radius: 50%;
        background: linear-gradient(135deg, #E0B43A 0%, #C9A227 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1rem auto;
        box-shadow: 0 4px 18px rgba(201, 162, 39, 0.35);
    }
    .hero h1 {
        color: #E8C15A;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.2px;
    }
    .hero p {
        color: #D7CCC8;
        margin-top: 0.5rem;
        font-size: 0.98rem;
    }
    h1, h2, h3 {
        color: #E8C15A !important;
    }
    .stInfo {
        background-color: rgba(201, 162, 39, 0.12) !important;
        border-left: 4px solid #C9A227 !important;
        color: #FFF8E1 !important;
    }
    label, .stMarkdown p {
        color: #FFF8E1 !important;
    }
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background-color: #3E2723 !important;
        border: 1px solid #8D6E63 !important;
        color: #FFF8E1 !important;
    }
    div.stButton > button[kind="primary"] {
        background-color: #C9A227;
        color: #2B1D16;
        border: none;
        font-weight: 700;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        width: 100%;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #E0B43A;
        color: #2B1D16;
    }
    .stSuccess {
        background-color: rgba(201, 162, 39, 0.15) !important;
        border-left: 4px solid #C9A227 !important;
        color: #FFF8E1 !important;
    }
    .stCaption, small {
        color: #BCAAA4 !important;
    }
    hr {
        border-color: #6D4C41 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# STEP 3: Load the trained model (path is relative to this file, not the terminal's folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "delivery_model.pkl")


@st.cache_resource
def load_model(model_path):
    with open(model_path, "rb") as file:
        package = pickle.load(file)
    return package["model"], package["features"]


if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found: {MODEL_PATH}\n\n"
        "Place 'delivery_model.pkl' in the same folder as app.py."
    )
    st.stop()

model, features = load_model(MODEL_PATH)

# STEP 4: Display the header
st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="46" height="46"
                 viewBox="0 0 24 24" fill="none" stroke="#2B1D16"
                 stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <path d="M3 7l9-4 9 4-9 4-9-4z"/>
                <path d="M3 7v10l9 4 9-4V7"/>
                <path d="M12 11v10"/>
            </svg>
        </div>
        <h1>Food Delivery Time Predictor</h1>
        <p>Estimate food delivery time using a trained Linear Regression model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info("Enter the details of a new order to estimate its delivery duration.")

# STEP 5: Collect delivery information
st.subheader("Enter Delivery Details")

distance_km = st.number_input(
    "Delivery distance (km)", min_value=0.1, max_value=50.0, value=5.0, step=0.5
)
preparation_time_min = st.number_input(
    "Food preparation time (minutes)", min_value=1, max_value=120, value=15
)
traffic_level = st.selectbox(
    "Traffic level",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "1 - Low",
        2: "2 - Moderate",
        3: "3 - High",
        4: "4 - Severe",
    }[x],
)
weather_severity = st.selectbox(
    "Weather severity",
    options=[0, 1, 2, 3],
    format_func=lambda x: {
        0: "0 - Clear",
        1: "1 - Light adverse weather",
        2: "2 - Bad weather",
        3: "3 - Severe weather",
    }[x],
)
rider_experience_years = st.number_input(
    "Rider experience (years)", min_value=0, max_value=50, value=3
)

# STEP 6: Predict when the user clicks the button
if st.button("Predict Delivery Time", type="primary"):
    new_delivery = pd.DataFrame(
        [[
            distance_km,
            preparation_time_min,
            traffic_level,
            weather_severity,
            rider_experience_years,
        ]],
        columns=features,
    )

    prediction = model.predict(new_delivery)[0]
    prediction = max(0, prediction)  # prevent a negative duration

    st.success(f"Estimated delivery time: {prediction:.1f} minutes")
    st.caption("This is a model estimate, not a guaranteed arrival time.")

st.divider()
