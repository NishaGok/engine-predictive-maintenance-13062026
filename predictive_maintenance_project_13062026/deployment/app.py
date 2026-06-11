
import streamlit as st
import pandas as pd
import joblib

from huggingface_hub import hf_hub_download

# --------------------------------------------------
# Load Model
# --------------------------------------------------

model_path = hf_hub_download(
    repo_id="NishaGok/engine-predictive-maintenance-model-13062026",
    filename="best_engine_predictive_model.pkl"
)

model = joblib.load(model_path)

# --------------------------------------------------
# UI
# --------------------------------------------------

st.title(
    "Predictive Maintenance System"
)

st.write(
    """
    Predict whether an engine
    requires maintenance based on
    sensor readings.
    """
)

engine_rpm = st.number_input(
    "Engine RPM",
    value=800
)

lub_oil_pressure = st.number_input(
    "Lubricating Oil Pressure",
    value=3.0
)

fuel_pressure = st.number_input(
    "Fuel Pressure",
    value=6.0
)

coolant_pressure = st.number_input(
    "Coolant Pressure",
    value=2.0
)

lub_oil_temp = st.number_input(
    "Lubricating Oil Temperature",
    value=77.0
)

coolant_temp = st.number_input(
    "Coolant Temperature",
    value=78.0
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "Predict Engine Condition"
):

    input_df = pd.DataFrame([{

        "engine_rpm": engine_rpm,

        "lub_oil_pressure": lub_oil_pressure,

        "fuel_pressure": fuel_pressure,

        "coolant_pressure": coolant_pressure,

        "lub_oil_temp": lub_oil_temp,

        "coolant_temp": coolant_temp

    }])

    prediction = model.predict(
        input_df
    )[0]

    probability = model.predict_proba(
        input_df
    )[0][1]

    if prediction == 1:

        st.error(
            f"""
            Maintenance Required

            Risk Probability:
            {probability:.2%}
            """
        )

    else:

        st.success(
            f"""
            Engine Operating Normally

            Confidence:
            {(1-probability):.2%}
            """
        )
