#the initial page for the streamlit app
import streamlit as st
from matplotlib import pyplot as plt
import json
import numpy as np
from acne_model import data_parsing, model_building
from acne_model.model import state_evolution_vv as evolution_function
from acne_model.model import reparameterize
from acne_model.model import map_latent_states_to_severity_probs as severity_distribution_function
from acne_model.model import latent_state_clustering

st.title("Acne Severity Analysis")

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
    st.write(this_built_model)
    return this_built_model, these_averages
def build_trajectory(**kwargs):
    """Function to create a new treatment trajectory given the everpresent widgets above."""

    #parse out treatment ends into full history

    if st.session_state.commit_requested:
        # Log current slider values as events if not already in draft_trajectory
        st.session_state.commit_requested = False

        # Then build trajectory
        full_history = []
        for index, end_history in enumerate(st.session_state.draft_trajectory):
            last_piece = [] if len(full_history) == 0 else full_history[-1]
            if index == 0:
                #last_treatment_list = full_history[]
                for day in range(1, end_history[1] + 1):
                    history_piece = [(end_history[0], day)] #list of tuples
                    full_history.append(history_piece)
                    #last_piece = tuple(history_piece)

            else:
                for day in range(1, end_history[1] + 1):
                    history_piece = (end_history[0], day)  # list of tuples
                    correct_history_piece = last_piece + [history_piece]
                    full_history.append(correct_history_piece)
                #last_piece = correct_history_piece

        st.session_state.trajectories.append(full_history)
        st.session_state.draft_trajectory.clear()
        st.session_state.prev_retinol = 0
        st.session_state.prev_antibiotics = 0
        st.session_state._reset_sliders = True
        st.session_state.debug_last_commit = full_history

        return full_history
def compute_severity_series(fitted_model_params, severity_deltas, scoring_hyperparams, trajectory, initial_state):
    """"Function using the fitted model params to predict a severity series for a given treatment trajectory."""
    last_latent_state = initial_state

    #FOR TESTING PURPOSES, TSTD is a frozen hardcoded value. In the future, it'll be returned from the training data

    expected_severity_changes = []
    for index, history_day in enumerate(trajectory):
        reparams = reparameterize(fitted_model_params)

        output_latent_state, unused_tstd, days_antibiotics, cream_used = evolution_function(v_t_last = last_latent_state, params = reparams, raw_distribution = None, severity_deltas = severity_deltas,
                                    index = index, history = history_day)

        this_severity_distribution = severity_distribution_function(output_latent_state, scoring_hyperparams)
        #computing expected change, given the distribution as a simple mean
        severities_deltas_array = np.array(list(severity_deltas.values()))

        expected_severity_change = np.average(a=severities_deltas_array, weights=this_severity_distribution, axis = 0)
        expected_severity_changes.append(expected_severity_change)

        #finding uncertainity bands
        #correct_posterior =

        #st.write("expected severity change", expected_severity_change)
def antibiotics_changed():
    st.session_state.draft_trajectory.append(
        ("Antibiotics", st.session_state.draft_antibiotics)
    )
def retinol_changed():
    st.session_state.draft_trajectory.append(
        ("Cream", st.session_state.draft_retinol)
    )
def commit_trajectory():
    st.session_state.trajectories.append(
        list(st.session_state.draft_trajectory)
    )
    st.session_state.draft_trajectory.clear()

    # reset sliders
    st.session_state.draft_antibiotics = 0
    st.session_state.draft_retinol = 0
def main():
    """
    """

    # Clinician Input Sliders
    st.header("Treatment Inputs")

    # initialization
    if "trajectories" not in st.session_state:
        st.session_state.trajectories = []

    if "draft_antibiotics" not in st.session_state:
        st.session_state.draft_antibiotics = 0

    if "prev_antibiotics" not in st.session_state:
        st.session_state.prev_antibiotics = 0

    if "draft_retinol" not in st.session_state:
        st.session_state.draft_retinol = 0

    if "prev_retinol" not in st.session_state:
        st.session_state.prev_retinol = 0

    if "draft_trajectory" not in st.session_state:
        st.session_state.draft_trajectory = []

    if st.session_state.get("_reset_sliders", False):
        st.session_state.draft_antibiotics = 0
        st.session_state.draft_retinol = 0
        st.session_state._reset_sliders = False

    if "commit_requested" not in st.session_state:
        st.session_state.commit_requested = False

    initial_inflammatory_state = st.slider("Inflammation", 0.0, 1.0, 0.1)
    initial_sebum = st.slider("Sebum", 0.0, 1.0, 0.1)
    initial_bacterial_load = st.slider("Bacterial Load", 0.0, 1.0, 0.1)

    initial_inflamatory_state = st.number_input("Patient's Initial Severity")
    initial_baseline_severity = st.number_input("Patient's Baseline Severity")

    antibiotics_days = st.slider(
    "Days of Antibiotics",
    0, 30,
    value=0 if st.session_state.get("_reset_sliders", False) else st.session_state.draft_antibiotics,
    key="draft_antibiotics",
    on_change=antibiotics_changed())

    retinol_days = st.slider(
    "Days of Retinol",
    0, 30,
    value=0 if st.session_state.get("_reset_sliders", False) else st.session_state.draft_retinol,
    key="draft_retinol",
    on_change=retinol_changed)

    st.session_state._reset_sliders = False

    configured_initial_state = np.array([initial_bacterial_load, initial_inflammatory_state, initial_sebum])

    if st.button("Commit trajectory"):
        st.session_state.commit_requested = True

    build_trajectory()

    this_called_model, these_deltas = calling_model("sim_acne.csv", "initial_constants.json")

    for trajectory in st.session_state.trajectories:
        computed_severity_series = compute_severity_series(fitted_model_params=this_called_model[1], severity_deltas=these_deltas,
            trajectory=trajectory, scoring_hyperparams=this_called_model[2], initial_state=configured_initial_state)

    these_inferred_latent_states = list(this_called_model[0].values())
    clustered_states = latent_state_clustering(these_inferred_latent_states)
    st.write("states", clustered_states)

    fig, ax = plt.subplots()
    #ax.scatter(antibiotics_days, retinol_days)
    # other plotting actions...
    st.pyplot(fig)




if __name__ == "__main__":
    main()
