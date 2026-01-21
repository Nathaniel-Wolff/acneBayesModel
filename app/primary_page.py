#the initial page for the streamlit app
import streamlit as st
from matplotlib import pyplot as plt
import json
import numpy as np
from acne_model import data_parsing, model_building
from scipy.stats import beta
from acne_model.model import state_evolution_vv as evolution_function
from acne_model.model import reparameterize
from acne_model.model import map_latent_states_to_severity_probs as severity_distribution_function
from acne_model.model import compute_cluster_dirichlets, assign_label
from collections import defaultdict

st.title("Acne Severity Analysis")
st.set_page_config(layout="wide")

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
    st.write(this_built_model[1])
    return this_built_model, these_averages, these_dirichlets
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
def compute_severity_series(fitted_model_params, severity_deltas, scoring_hyperparams, trajectory, initial_state, centers, labels, dirichlets):
    """"Function using the fitted model params to predict a latent state for a given treatment trajectory."""
    last_latent_state = initial_state
    pred_latent_states = []
    dirs_and_expectations = defaultdict(tuple)
    histories = []

    #collecting Dirichlets
    for index, history_day in enumerate(trajectory):
        reparams = reparameterize(fitted_model_params)
        output_latent_state, unused_tstd, days_antibiotics, cream_used = evolution_function(v_t_last = last_latent_state, params = reparams, raw_distribution = None, severity_deltas = severity_deltas,
                                    index = index, history = history_day)

        label = assign_label(output_latent_state, centers)
        actual_label = str(labels[label])
        fetched_dirichlet = dirichlets[actual_label]
        pred_latent_states.append(output_latent_state)

        this_severity_distribution = severity_distribution_function(output_latent_state, scoring_hyperparams)
        #computing expected change, given the distribution as a simple mean
        severities_deltas_array = np.array(list(severity_deltas.values()))

        expected_severity_change = np.average(a=severities_deltas_array, weights=this_severity_distribution, axis = 0)
        dirs_and_expectations[index] = (expected_severity_change, fetched_dirichlet)
        histories.append(history_day)

        last_latent_state = output_latent_state

    return dirs_and_expectations, histories


def plot_severity_series(plot, axis, trajectories_dirichlets, severity_deltas, window_size = 20, confidence_level = .95, width = 1):
    """Function to plot Dirichlet credible intervals over a plot."""
    colors = ["tab:green", "tab:gray", "tab:red"]
    labels = ["High Decrease", "Medium Decrease", "Low Decrease"]

    day = st.session_state.day
    initial_severity = st.session_state.initial_severity

    start_limit = day
    end_limit = start_limit + window_size

    real_deltas = list(severity_deltas.values())

    for trajectory in trajectories_dirichlets:
        T = len(trajectory)
        t = np.arange(T)

        means = {k: [] for k in range(3)}
        low = {k: [] for k in range(3)}
        high = {k: [] for k in range(3)}
        for index, (expected_val, dirichlet) in trajectory.items():
            alpha = np.array(dirichlet)
            alpha0 = alpha.sum()

            for k in range(3):
                means[k].append(alpha[k] / alpha0)
                low[k].append(beta.ppf(0.025, alpha[k], alpha0 - alpha[k]))
                high[k].append(beta.ppf(0.975, alpha[k], alpha0 - alpha[k]))

        for k, (label, color) in enumerate(zip(labels, colors)):
            real_delta = real_deltas[k]
            # *np.ones(len(means[k][start_limit:end_limit]))*real_delta
            plot.plot(t[start_limit:end_limit], (means[k][start_limit:end_limit]), color=color, label=label)
            plot.fill_between(
                t[start_limit:end_limit],
                (low[k][start_limit:end_limit]),#*np.ones(len(low[k][start_limit:end_limit]))*real_delta,
                (high[k][start_limit:end_limit]),#*np.ones(len(high[k][start_limit:end_limit]))*real_delta,
                color=color,
                alpha=0.25
            )

        plot.set_xlim(start_limit, end_limit)
        #plot.set_ylim(0, 1)
        plot.set_xlabel("Day")
        plot.set_ylabel("Acne Severity Percentage")
        plot.legend()

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # Plot the legend using only the unique handles and labels
    ax = plt.gca()  # Get the current axis
    ax.relim()  # Recompute the data limits
    ax.autoscale_view(True, True, True)  # Apply the new limits to all axes
    plt.legend(by_label.values(), by_label.keys())

def baseline_sev_changed():
    st.session_state.baseline_severity = st.session_state.draft_severity
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

def set_window():
    pass
def main():
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

    if "these_deltas" not in st.session_state:
        st.session_state.these_deltas = {}

    if st.session_state.get("_reset_sliders", False):
        st.session_state.draft_antibiotics = 0
        st.session_state.draft_retinol = 0
        st.session_state._reset_sliders = False

    if "commit_requested" not in st.session_state:
        st.session_state.commit_requested = False

    if "day" not in st.session_state:
        st.session_state.day = 0
    if "window_size" not in st.session_state:
        st.session_state.window_size = 10
    if "T" not in st.session_state:
        st.session_state.T= 20

    if "draft_severity" not in st.session_state:
        st.session_state.draft_severity = 1.0

    if "initial_severity" not in st.session_state:
        st.session_state.initial_severity = 1.0


    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Previous"):
            st.session_state.day = max(0, st.session_state.day - 1)
    with col2:
        if st.button("Next ➡️"):
            st.session_state.day = min(st.session_state.T - st.session_state.window_size, st.session_state.day + 1)

    initial_inflammatory_state = st.slider("Inflammation", 0.0, 1.0, 0.1)
    initial_sebum = st.slider("Sebum", 0.0, 1.0, 0.1)
    initial_bacterial_load = st.slider("Bacterial Load", 0.0, 1.0, 0.1)

    #initial_inflamatory_state = st.number_input("Patient's Initial Severity")
    initial_baseline_severity = st.number_input("Patient's Initial Severity", help = "(Pre-Normalize for best results)", key = "draft_severity", on_change = baseline_sev_changed())


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

    selected_window_size = st.number_input("Window Size", min_value = 1, width = "stretch")
    st.session_state.window_size = selected_window_size

    st.session_state._reset_sliders = False

    configured_initial_state = np.array([initial_bacterial_load, initial_inflammatory_state, initial_sebum])

    if st.button("Commit trajectory"):
        st.session_state.commit_requested = True

    build_trajectory()

    this_called_model, these_deltas, observed_dirichlets = calling_model("sim_acne.csv", "initial_constants.json")
    st.session_state.these_deltas = these_deltas

    these_inferred_latent_states = list(this_called_model[0].values())
    clustered_states, these_clusters = compute_cluster_dirichlets(these_inferred_latent_states, observed_dirichlets)
    these_new_centers, these_labels, these_counts, these_regions_distances = these_clusters

    all_severity_series = []
    for trajectory in st.session_state.trajectories:
        computed_severity_series, histories = compute_severity_series(fitted_model_params=this_called_model[1], severity_deltas=these_deltas,
            trajectory=trajectory, scoring_hyperparams=this_called_model[2], initial_state=configured_initial_state, centers = these_new_centers, labels = these_labels,
                                                           dirichlets=clustered_states)
        all_severity_series.append(computed_severity_series)

    col1, col2 = st.columns([20, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(20, 8))
        plotted_trajectories = plot_severity_series(ax, fig, all_severity_series, st.session_state.these_deltas,
                                                    window_size=st.session_state.window_size, confidence_level=.95)
        st.write(all_severity_series) #testing
        st.pyplot(fig)

    with col2:
        st.write(".")



    #st.pyplot(fig, width = "stretch")

if __name__ == "__main__":
    main()
