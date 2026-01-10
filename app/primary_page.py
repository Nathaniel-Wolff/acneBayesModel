#the initial page for the streamlit app
import streamlit as st
from matplotlib import pyplot as plt
import json
import numpy as np
from acne_model import data_parsing, model_building


st.title("Acne Severity Analysis")

#Clinician Inputs
st.header("Treatment Inputs")
antibiotics_days = st.slider("Days of Antibiotics", 10, 30, 10)
retinol_days = st.slider("Days of Retinol", 10, 30, 10)

initial_inflammatory_state = st.slider("Inflammation", 0.0, 1.0, 0.1)
initial_sebum  = st.slider("Sebum", 0.0, 1.0, 0.1)
initial_bacterial_load = st.slider("Bacterial Load", 0.0, 1.0, 0.1)

initial_inflamatory_state = st.number_input("Patient's Initial Severity")
initial_baseline_severity = st.number_input("Patient's Baseline Severity")

#calling the model with test dataset
def calling_model(this_raw_data_name, json_name):
    data_returns = data_parsing(this_raw_data_name)
    these_ranges = data_returns[3]
    these_averages = data_returns[5]
    these_dirichlets = data_returns[6]

    with open(json_name, "r") as icgs:
        initial_constant_guesses = json.load(icgs)
        this_model_config = {"scoring": np.random.randn(3, 3) * 0.02,
                             # column order: low severity change, medium, high. row order: #bacteria, inflammation, sebum.
                             "biases": [0, 0, 0],
                             "Q": np.eye(3) * (initial_constant_guesses["w_sigma"] ** 2),
                             "R": np.eye(3) * (initial_constant_guesses["m_sigma"] ** 2)}

    #reparameterization and the reverse are both done in the function itself now
    this_built_model = model_building(data_returns[1], initial_constant_guesses, these_dirichlets, these_averages,
                                      this_model_config)
    return this_built_model


this_called_model = calling_model("sim_acne.csv", "initial_constants.json")
st.write(this_called_model)


fig, ax = plt.subplots()
ax.scatter(antibiotics_days, retinol_days)
# other plotting actions...
st.pyplot(fig)




