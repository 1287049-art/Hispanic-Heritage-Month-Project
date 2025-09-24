import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import datetime

st.set_page_config(page_title="Latin America Historical Data Analysis", layout="wide")

st.title("Daniel Estrella's Latin America Data Regression App")

# ------------------------
# Helper Functions
# ------------------------
@st.cache_data
def get_data():
    # Historical data example (realistic but simplified for deployment)
    # You would replace these with real datasets from sources like World Bank or OWID
    years = np.arange(1950, 2024)
    
    data = {
        'Argentina': {
            'Population': np.linspace(17e6, 45e6, len(years)) + np.random.normal(0, 1e6, len(years)),
            'Unemployment rate': np.random.uniform(3, 12, len(years)),
            'Life expectancy': np.linspace(60, 77, len(years)) + np.random.normal(0, 1, len(years)),
        },
        'Brazil': {
            'Population': np.linspace(53e6, 214e6, len(years)) + np.random.normal(0, 2e6, len(years)),
            'Unemployment rate': np.random.uniform(2, 14, len(years)),
            'Life expectancy': np.linspace(55, 76, len(years)) + np.random.normal(0, 1, len(years)),
        },
        'Mexico': {
            'Population': np.linspace(28e6, 130e6, len(years)) + np.random.normal(0, 1.5e6, len(years)),
            'Unemployment rate': np.random.uniform(2, 10, len(years)),
            'Life expectancy': np.linspace(50, 75, len(years)) + np.random.normal(0, 1, len(years)),
        }
    }
    return years, data

def fit_poly_regression(x, y, degree=3):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x.reshape(-1,1), y)
    return model

def analyze_function(x, y, model):
    # Compute first derivative approx
    deriv = np.gradient(model.predict(x.reshape(-1,1)), x)
    max_idx = np.argmax(y)
    min_idx = np.argmin(y)
    fastest_incr_idx = np.argmax(deriv)
    fastest_decr_idx = np.argmin(deriv)
    analysis = {
        'Local max': (x[max_idx], y[max_idx]),
        'Local min': (x[min_idx], y[min_idx]),
        'Fastest increase': (x[fastest_incr_idx], deriv[fastest_incr_idx]),
        'Fastest decrease': (x[fastest_decr_idx], deriv[fastest_decr_idx]),
        'Domain': (x[0], x[-1]),
        'Range': (min(y), max(y))
    }
    return analysis

def format_analysis(analysis, category, country):
    s = f"### Analysis for {category} in {country}\n"
    s += f"- Local maximum: {analysis['Local max'][1]:.2f} at year {int(analysis['Local max'][0])}\n"
    s += f"- Local minimum: {analysis['Local min'][1]:.2f} at year {int(analysis['Local min'][0])}\n"
    s += f"- Fastest increase: {analysis['Fastest increase'][1]:.2f} per year at year {int(analysis['Fastest increase'][0])}\n"
    s += f"- Fastest decrease: {analysis['Fastest decrease'][1]:.2f} per year at year {int(analysis['Fastest decrease'][0])}\n"
    s += f"- Domain: {analysis['Domain'][0]} to {analysis['Domain'][1]}\n"
    s += f"- Range: {analysis['Range'][0]:.2f} to {analysis['Range'][1]:.2f}\n"
    return s

# ------------------------
# Main App
# ------------------------
years, data = get_data()

category = st.selectbox("Select Data Category", ['Population', 'Unemployment rate', 'Life expectancy'])
countries = st.multiselect("Select countries to display", list(data.keys()), default=['Argentina'])

year_step = st.slider("Graph increments (years)", 1, 10, 1)

degree = st.slider("Degree of polynomial regression", 3, 6, 3)

st.subheader("Raw Data Table (Editable)")
df_display = pd.DataFrame({country: data[country][category] for country in countries}, index=years)
edited_df = st.data_editor(df_display, num_rows="dynamic")

# ------------------------
# Regression & Plot
# ------------------------
st.subheader("Regression Graph")
plt.figure(figsize=(10,6))

for country in countries:
    x = years
    y = edited_df[country].values
    model = fit_poly_regression(x, y, degree)
    
    x_plot = np.arange(years[0], years[-1]+1, year_step)
    y_plot = model.predict(x_plot.reshape(-1,1))
    
    # Extrapolation
    extrap_years = st.number_input(f"Extrapolate years for {country}", min_value=0, max_value=50, value=0)
    if extrap_years > 0:
        x_extra = np.arange(years[-1]+1, years[-1]+extrap_years+1)
        y_extra = model.predict(x_extra.reshape(-1,1))
        plt.plot(x_extra, y_extra, '--', label=f"{country} (Extrapolated)")
        x_plot = np.concatenate([x_plot, x_extra])
        y_plot = np.concatenate([y_plot, y_extra])
    
    plt.scatter(x, y, label=f"{country} Data")
    plt.plot(x_plot, y_plot, label=f"{country} Regression")

plt.xlabel("Year")
plt.ylabel(category)
plt.legend()
plt.grid(True)
st.pyplot(plt)

# ------------------------
# Display Equation
# ------------------------
st.subheader("Regression Equations")
for country in countries:
    x = years
    y = edited_df[country].values
    model = fit_poly_regression(x, y, degree)
    coeffs = model.named_steps['linearregression'].coef_
    intercept = model.named_steps['linearregression'].intercept_
    equation = " + ".join([f"{coeff:.3e}*x^{i}" for i, coeff in enumerate(coeffs)])
    st.text(f"{country}: y = {intercept:.3e} + {equation}")

# ------------------------
# Function Analysis
# ------------------------
st.subheader("Function Analysis")
for country in countries:
    x = years
    y = edited_df[country].values
    model = fit_poly_regression(x, y, degree)
    analysis = analyze_function(x, y, model)
    st.markdown(format_analysis(analysis, category, country))

# ------------------------
# Prediction / Extrapolation
# ------------------------
st.subheader("Prediction / Interpolation / Extrapolation")
selected_country = st.selectbox("Select country for prediction", countries)
input_year = st.number_input("Input year", int(years[0]), int(years[-1]+50), value=int(years[-1]+1))
model = fit_poly_regression(years, edited_df[selected_country].values, degree)
predicted_value = model.predict(np.array([[input_year]]))[0]
st.write(f"Predicted {category} for {selected_country} in {int(input_year)}: {predicted_value:.2f}")

# ------------------------
# Average Rate of Change
# ------------------------
st.subheader("Average Rate of Change")
y1 = st.number_input("Start year", int(years[0]), int(years[-1]), value=int(years[0]))
y2 = st.number_input("End year", int(years[0]), int(years[-1]+50), value=int(years[-1]))
if y2 > y1:
    v1 = model.predict(np.array([[y1]]))[0]
    v2 = model.predict(np.array([[y2]]))[0]
    avg_rate = (v2 - v1)/(y2 - y1)
    st.write(f"Average rate of change of {category} for {selected_country} between {int(y1)} and {int(y2)}: {avg_rate:.2f} per year")
