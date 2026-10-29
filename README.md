<h1 align="center">Food Delivery Time Predictor</h1>

<p align="center">
  A machine learning project that estimates food delivery time using Linear Regression, with a command-line interface and an interactive Streamlit web application.
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pandas.pydata.org/"><img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"></a>
  <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn"></a>
  <a href="https://matplotlib.org/"><img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"></a>
  <a href="https://jupyter.org/"><img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter"></a>
  <a href="https://docs.python.org/3/library/pickle.html"><img src="https://img.shields.io/badge/Pickle-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Pickle"></a>
  <a href="https://scikit-learn.org/stable/modules/linear_model.html"><img src="https://img.shields.io/badge/Linear%20Regression-0A66C2?style=for-the-badge" alt="Linear Regression"></a>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [UI Design](#ui-design)
- [Pipeline Outputs](#pipeline-outputs)
- [Input Ranges](#input-ranges)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

Estimating how long a food delivery will take is a common problem for food delivery platforms and customers. This project builds an end-to-end pipeline that:

- Generates a historical delivery dataset
- Explores the data and examines the relationship between features and delivery time
- Trains a Linear Regression model and evaluates it on held-out data
- Saves the trained model for reuse
- Serves predictions through a command-line script and an interactive web application

The model predicts delivery time in minutes from five inputs: distance, food preparation time, traffic level, weather severity, and rider experience.

## Features

- Synthetic dataset generation with 1,500 delivery records
- Exploratory data analysis with summary statistics, correlations, and a distance-versus-time scatter plot
- Model training with an 80/20 train-test split and reproducible results (fixed random seed)
- Evaluation using MAE, MSE, RMSE, and R-squared
- Model persistence using pickle, storing the model together with its feature order
- Command-line prediction with input validation
- Streamlit web application with a custom themed interface and dropdown-based inputs

## Project Structure

```text
food_delivery_app/
|-- 01_create_data.py          # Generates the synthetic delivery dataset
|-- 02_explore_data.py         # Exploratory data analysis and visualization
|-- 02_explore_data.ipynb      # Jupyter Notebook version of the EDA
|-- 03_train_model.py          # Trains and evaluates the model, saves it to disk
|-- 04_predict.py              # Command-line prediction tool
|-- app.py                     # Streamlit web application
|-- delivery_data_Tandrima.csv # Generated dataset (created by 01_create_data.py)
|-- delivery_model.pkl         # Saved model and feature list (created by 03_train_model.py)
|-- requirements.txt           # Project dependencies
|-- README.md                  # Project documentation
|-- Outputs.md                 # Recorded outputs of each step
`-- images/                    # Screenshots used in this README
```

## Technology Stack

- Python: core programming language
- Pandas: data loading, manipulation, and summary statistics
- NumPy: numerical generation and computation
- scikit-learn: Linear Regression model and evaluation metrics
- Matplotlib: data visualization
- Streamlit: interactive web interface
- Jupyter Notebook: exploratory analysis documentation
- Pickle: model serialization

## Methodology

### 1. Data Generation

`01_create_data.py` generates 1,500 synthetic delivery records using a fixed random seed (42) so results are reproducible. The input features are:

| Feature | Description | Range |
|---|---|---|
| `distance_km` | Delivery distance in kilometers | 1 to 15 |
| `preparation_time_min` | Food preparation time in minutes | 5 to 30 |
| `traffic_level` | Traffic intensity (categorical scale) | 1 to 4 |
| `weather_severity` | Weather condition severity | 0 to 3 |
| `rider_experience_years` | Rider experience in years | 0 to 10 |
| `delivery_time_min` | Target: total delivery time in minutes | Derived |

The target variable is generated from the following relationship, with a small random noise term added:

```text
delivery_time = 5 + 3 * distance + preparation_time + 4 * traffic_level
                + 2 * weather_severity - 0.4 * rider_experience + noise
```

### 2. Exploratory Data Analysis

`02_explore_data.py` (and its notebook version) inspects the dataset shape, data types, missing values, and descriptive statistics. It also computes correlations with the target variable and plots distance against delivery time.

### 3. Model Training

`03_train_model.py` performs the following steps:

- Loads the dataset and selects the five input features
- Splits the data into 1,200 training records and 300 testing records
- Trains a `LinearRegression` model
- Evaluates predictions on the test set
- Saves the model and the ordered feature list to `delivery_model.pkl`

### 4. Prediction

Predictions can be made in two ways:

- `04_predict.py`: prompts the user for each input in the terminal, validates the values, and prints the estimated delivery time
- `app.py`: provides a graphical form in the browser and displays the estimate when the user clicks the prediction button

## Model Performance

The model was evaluated on 300 held-out test records.

| Metric | Value | Interpretation |
|---|---|---|
| MAE | 2.51 minutes | On average, predictions differ from actual values by about 2.5 minutes |
| MSE | 9.63 | Mean squared error |
| RMSE | 3.10 minutes | Typical prediction error, penalizing larger errors more |
| R-squared | 0.9649 | The model explains about 96.5% of the variance in delivery time |

### Learned Coefficients

| Feature | Coefficient |
|---|---|
| `distance_km` | 3.007 |
| `preparation_time_min` | 0.997 |
| `traffic_level` | 3.980 |
| `weather_severity` | 1.979 |
| `rider_experience_years` | -0.358 |
| Intercept | 4.941 |

The resulting model equation is:

```text
delivery_time = 4.941 + 3.007 * distance_km + 0.997 * preparation_time_min
                + 3.980 * traffic_level + 1.979 * weather_severity
                - 0.358 * rider_experience_years
```

Each coefficient shows the expected change in delivery time (in minutes) for a one-unit increase in that feature, holding the others constant. The negative coefficient for rider experience indicates that more experienced riders are associated with shorter delivery times.

## Installation

### Prerequisites

- Python 3.9 or later
- pip

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/food-delivery-time-predictor.git
   cd food-delivery-time-predictor
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv food_delivery_app
   source food_delivery_app/bin/activate
   ```

   On Windows:

   ```bash
   food_delivery_app\Scripts\activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the following commands in order from the project folder. Keep all files in the same directory.

```bash
# Step 1: Generate the dataset
python 01_create_data.py

# Step 2 (optional): Run exploratory analysis
python 02_explore_data.py

# Step 3: Train the model and save it
python 03_train_model.py

# Step 4: Make a prediction from the command line
python 04_predict.py

# Step 5: Launch the web application
streamlit run app.py
```

After running `streamlit run app.py`, open the local address shown in the terminal (typically `http://localhost:8501`).

Note: `delivery_data_Tandrima.csv` and `delivery_model.pkl` are generated by the first two scripts. If they are already included in the repository, you can skip directly to the step you need.

## UI Design

The Streamlit application provides a single-page form for entering delivery details. The interface uses a brown and golden yellow theme with the following components:

- Numeric inputs for distance, preparation time, and rider experience
- Dropdown menus with descriptive labels for traffic level and weather severity
- A primary action button to generate the prediction
- A result panel showing the estimated delivery time, with a note that the value is a model estimate and not a guaranteed arrival time

![Food Delivery Time Predictor Interface]
<img width="1920" height="1080" alt="Screenshot (68)" src="https://github.com/user-attachments/assets/a62f7fa1-75b1-489b-9f28-14b8331bd9d8" />
<img width="1920" height="1080" alt="Screenshot (69)" src="https://github.com/user-attachments/assets/109874b3-804e-4760-9960-33467de1cce2" />
<img width="1920" height="1080" alt="Screenshot (71)" src="https://github.com/user-attachments/assets/71bacbe8-0d55-4211-b538-c3c2939ba4d0" />




## Pipeline Outputs

The terminal outputs from each stage of the pipeline are summarized below. The complete records are also available in `Outputs.md`.

**Dataset generation (`01_create_data.py`)**

![Dataset Generation](images/01_create_data.png)

**Model training (`03_train_model.py`)**

![Model Training](images/03_train_model.png)

**Command-line prediction (`04_predict.py`)**

![Command-line Prediction](images/04_predict.png)

## Input Ranges

The model was trained on the following ranges:

| Input | Training Range | Accepted in App |
|---|---|---|
| Distance | 1 to 15 km | 0.1 to 50 km |
| Preparation time | 5 to 30 min | 1 to 120 min |
| Traffic level | 1 to 4 | 1 to 4 |
| Weather severity | 0 to 3 | 0 to 3 |
| Rider experience | 0 to 10 years | 0 to 50 years |

Predictions made outside the training range are extrapolations and may be less reliable.

## Limitations

- The dataset is synthetic and generated from a known formula. The high R-squared value reflects the linear structure of the generation process and may not represent real-world performance.
- Real delivery times depend on factors not included in this model, such as restaurant congestion, rider availability, order batching, and road-specific conditions.
- Linear Regression assumes a linear relationship between the inputs and the target. Non-linear effects are not captured.
- The model file depends on the scikit-learn version used to create it. If loading fails, re-run `03_train_model.py` in your environment.
- Scripts that read or write files use paths relative to the working directory or the script location, so run them from the project folder.

## Future Improvements

- Train on real-world delivery data collected from a food delivery platform
- Compare Linear Regression with tree-based models such as Random Forest and Gradient Boosting
- Add cross-validation and hyperparameter tuning
- Include additional features such as time of day, day of week, and restaurant location
- Add input validation and warnings in the web application for values outside the training range
- Containerize the application with Docker and deploy it to a cloud platform

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

**Tandrima**

Educational project | Python | Pandas | scikit-learn | Pickle | Streamlit
