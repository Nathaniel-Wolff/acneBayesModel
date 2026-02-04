#the initial page for the streamlit app
import streamlit as st
from matplotlib import pyplot as plt
import json
import numpy as np
from acne_model import data_parsing, model_building
from scipy.stats import beta
from acne_model.model import state_evolution_vv as evolution_function
from acne_model.model import reparameterize
from acne_model.model import map_latent_states_to_probs as severity_distribution_function
from acne_model.model import compute_cluster_dirichlets, assign_label, adjust_empirical_kernel
from collections import defaultdict

st.title("Acne Severity Analysis")
st.set_page_config(layout = "wide")

#calling the model with test dataset
def calling_model(this_raw_data_name, json_name):
    data_returns = data_parsing(this_raw_data_name)

    #these_ranges = data_returns[3]
    these_averages = data_returns[5]
    these_dirichlets = data_returns[6]
    these_empirical_trans_matrices = data_returns[7]
    transition_counts = data_returns[8]

    #st.write("empirical matrices", these_empirical_trans_matrices)



    with open(json_name, "r") as icgs:
        initial_constant_guesses = json.load(icgs)
        #change config later
        this_model_config = {"scoring": np.random.randn(3, 3) * 0.02,
                             # column order: low severity change, medium, high. row order: #bacteria, inflammation, sebum.
                             "biases": [0, 0, 0],
                             "scoring_trans_row 0": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                             "biases_trans_row 0": [0.0, 0.0, 0.0],
                             "scoring_trans_row 1": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                             "biases_trans_row 1": [0.0, 0.0, 0.0],
                             "scoring_trans_row 2": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                             "biases_trans_row 2": [0.0, 0.0, 0.0],
                             "Q": np.eye(3) * (initial_constant_guesses["w_sigma"] ** 2),
                             "R": np.eye(3) * (initial_constant_guesses["m_sigma"] ** 2)}


    #reparameterization and the reverse are both done in the function itself now
    this_built_model = model_building(these_empirical_counts=transition_counts, these_empirical_kernels=these_empirical_trans_matrices, these_initial_params=initial_constant_guesses, raw_distributions=these_dirichlets,
                                      severity_deltas=these_averages,
                                      model_config=this_model_config)

    return this_built_model, these_averages, these_dirichlets, these_empirical_trans_matrices, transition_counts
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
def compute_severity_series(fitted_model_params, severity_deltas, scoring_hyperparams, trajectory, initial_state, centers, labels, dirichlets, empirical_kernels):
    """"Function using the fitted model params to predict a latent state for a given treatment trajectory."""
    last_latent_state = initial_state
    pred_latent_states = []
    dirs_and_expectations = defaultdict(tuple)
    histories = []
    adjusted_kernels = []


    #collecting Dirichlets and finding adjusted Kernels
    for index, history_day in enumerate(trajectory):
        reparams = reparameterize(fitted_model_params)
        output_latent_state, unused_tstd, days_antibiotics, cream_used = evolution_function(v_t_last = last_latent_state, params = reparams, raw_distribution = None, severity_deltas = severity_deltas,
                                    index = index, history = history_day)

        adjusted_kernel = adjust_empirical_kernel(scoring_hyperparams=scoring_hyperparams, latent_state=output_latent_state, empirical_kernel=empirical_kernels[index+1])
        adjusted_kernels.append(adjusted_kernel)


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

    return dirs_and_expectations, histories, adjusted_kernels
def compute_subtrajectory_prob(trajectory, current_transition_matrix):
    last_pr = trajectory[1] #trajectory is in the form ([S1, S2...Sn], Pr(Trajectory))
    states_list = trajectory[0]

    if len(states_list) > 1:
        previous_state = states_list[len(states_list)-2]
        current_state = states_list[len(states_list)-1]
        current_transition_prob = current_transition_matrix[current_state][previous_state]
        last_pr *= current_transition_prob
    else:
        pass


    return last_pr
def extend_and_prune_trajectories(actual_trajectories, the_transition_matrices, the_top_log_pr, the_max_trajectories, initial_distribution):
    #1 for low change, 2 for medium change, 3 for high change
    if len(actual_trajectories) != 0:
        #------ extension
        extended_trajectories = []
        for traj, log_prob in actual_trajectories: #trajectory is in the form ([S1, S2...Sn], Pr(Trajectory))
            for next_state in [0, 1, 2]:
                new_traj = traj + [next_state]  # creates a new list
                extended_trajectories.append((new_traj, log_prob))
            actual_trajectories = extended_trajectories

        #------ probability calculation and pruning
        updated_trajectories = []
        for appended_trajectory in actual_trajectories:
            new_computed_prob_log = np.log(compute_subtrajectory_prob(appended_trajectory, the_transition_matrices) + 1e-12) #included small epsilon
            updated_trajectories.append((appended_trajectory[0], new_computed_prob_log))
        #---- sorting trajectories by probability
        updated_trajectories.sort(key=lambda x: x[1], reverse=True)
        #datatype = np.dtype([("traj", list), ("logprob", float)]) #datatype for these arrays
        #structured_trajectory_array = np.array(updated_trajectories, dtype = datatype)
        filtered_trajectories = [(traj, logp) for traj, logp in updated_trajectories if logp >= the_top_log_pr]

        array_as_list = list(filtered_trajectories)
        return array_as_list

    else:
        actual_trajectories = [([m], initial_distribution[m]) for m in range(3)] #initialize probabilities from initial distribution
        return actual_trajectories

def beam_search_trajectories(
        trajectory,
    transition_matrices, severity_deltas, initial_severity = 1.0,
    initial_distribution=np.array([0.7, 0.1, 0.1]),
    top_k=3):
    """
    Iterative beam search for subtrajectory generation (sequence of acne severity change states), given the properly inferred kernels.
    Returns:
        List of (trajectory, logprob) tuples, top-K sub_trajectories after max_length steps
    """

    # Initialize sub_trajectories: ([state1], logprob)
    sub_trajectories = [([i], float(np.log(initial_distribution[i] + 1e-12))) for i in range(len(initial_distribution))]

    for treatment_step in range(len(trajectory)):
        new_trajectories = []

        transition_matrix = transition_matrices[treatment_step]
        for traj, logp in sub_trajectories:
            last_state = traj[-1]
            running_prob = logp
            for next_state in range(3):  # 3 possible states: 0,1,2
                new_traj = traj + [next_state]
                transition_log_p = np.log(transition_matrix[last_state][next_state])
                new_logp = float(running_prob + transition_log_p)
                new_trajectories.append((new_traj, new_logp))

            # sort top-K
        new_trajectories.sort(key=lambda x: x[1], reverse=True)
        sub_trajectories = new_trajectories[:top_k]

        if not new_trajectories:
            break
    severity_delta_values = list(severity_deltas.values())
    #converting trajectories into severity decrease series (need to pair this with initial latent state severity decreases)

    all_severity_series = []

    for built_trajectory, prob in sub_trajectories:
        severity_series = [initial_severity]
        for state in built_trajectory:
            severity_series.append(severity_series[-1] * severity_delta_values[state]*.01)

        all_severity_series.append(severity_series)


    return sub_trajectories, all_severity_series
def plot_mixture_dir_sev_change(plot, trajectories_dirichlets, kernels, severity_deltas, window_size = 20, confidence_level = .95, width = 1):
    """Function to plot Dirichlet mixture, given both inferred kernels and inferred Dirichlets from clustering."""

    colors = ["tab:green", "tab:gray", "tab:red"]
    labels = ["High Decrease", "Medium Decrease", "Low Decrease"]

    day = st.session_state.day


    start_limit = day
    end_limit = start_limit + window_size

    real_deltas = np.array(list(severity_deltas.values()))





    for trajectory in trajectories_dirichlets:
        T = len(trajectory)
        t = np.arange(T)


        means = {k: [] for k in range(3)}
        low = {k: [] for k in range(3)}
        high = {k: [] for k in range(3)}

        probs = []



        initial_dist = np.array(list(trajectory.values())[0][1])
        initial_dist_sum =  np.sum(initial_dist)
        initial_dist_expectation = initial_dist / initial_dist_sum


        last_latent_state_pr = initial_dist_expectation
        severity_series = [st.session_state.initial_severity]
        st.write(severity_series)

        pruned_trajectories, the_top_severities_series = beam_search_trajectories(transition_matrices=kernels, trajectory=trajectory,
                                                       severity_deltas=st.session_state.these_deltas, initial_severity=st.session_state.initial_severity,
                                                       initial_distribution=initial_dist)



        for index, (expected_val, dirichlet) in trajectory.items():
            alpha = np.array(dirichlet)
            alpha0 = alpha.sum()

            kernel = kernels[index]

            next_prob = kernel @ last_latent_state_pr

            next_prob /= np.sum(next_prob)

            probs.append(next_prob)
            expected_severity_change_percent = 1 - (.01 * (real_deltas.T @ next_prob))
            severity_series.append(severity_series[-1] * expected_severity_change_percent)
            last_latent_state_pr = next_prob




            for k in range(3):
                means[k].append(alpha[k] / alpha0)
                low[k].append(beta.ppf(0.025, alpha[k], alpha0 - alpha[k]))
                high[k].append(beta.ppf(0.975, alpha[k], alpha0 - alpha[k]))

        for k, (label, color) in enumerate(zip(labels, colors)):
            #plot.plot(t[start_limit:end_limit], (means[k][start_limit:end_limit]), color=color, label=label)
            plot.plot(t[start_limit:end_limit], (severity_series[start_limit:end_limit]), color=color, label=label)

        for one_series in the_top_severities_series:
            plot.plot(t[start_limit:end_limit], (one_series[start_limit:end_limit]), color="blue")


        plot.set_xlim(start_limit, end_limit)

        plot.set_xlabel("Day")
        plot.set_ylabel("Acne Severity")
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

    if "empirical_matrices" not in st.session_state:
        st.session_state.empirical_matrices = []

    if "empirical_counts" not in st.session_state:
        st.session_state.empirical_counts = []

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

    initial_severity = st.slider(
        "Patient's Initial Severity",
        0.0, 30.0,
        value = 1.0 if st.session_state.get("_reset_sliders", False) else st.session_state.draft_severity,
        key="draft_severity",
        on_change=baseline_sev_changed())

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

    this_called_model, these_deltas, observed_dirichlets, ob_empirical_matrices, ob_raw_counts = calling_model("sim_acne.csv", "initial_constants.json")
    st.session_state.these_deltas = these_deltas
    st.session_state.empirical_matrices = ob_empirical_matrices
    st.session_state.raw_counts = ob_raw_counts

    latent_states_empirical = list(this_called_model[0].values())

    clustered_states, these_clusters = compute_cluster_dirichlets(latent_states_empirical, observed_dirichlets)
    these_new_centers, these_labels, these_counts, these_regions_distances = these_clusters

    all_severity_series = []
    for trajectory in st.session_state.trajectories:
        computed_severity_series, histories, these_kernels = compute_severity_series(fitted_model_params=this_called_model[1], severity_deltas=these_deltas,
            trajectory=trajectory, scoring_hyperparams=this_called_model[2], initial_state=configured_initial_state, centers = these_new_centers, labels = these_labels,
                                                           empirical_kernels=st.session_state.empirical_matrices, dirichlets=clustered_states)
        all_severity_series.append(computed_severity_series)
        pruned_trajectories = beam_search_trajectories(transition_matrices=these_kernels, trajectory=trajectory, severity_deltas=st.session_state.these_deltas, top_k=6)
        fig, ax = plt.subplots(figsize=(20, 8))
        plotted_trajectories = plot_mixture_dir_sev_change(plot=ax, trajectories_dirichlets=all_severity_series,
                                                           kernels=these_kernels,
                                                           severity_deltas=st.session_state.these_deltas,
                                                           window_size=st.session_state.window_size,
                                                           confidence_level=.95)
        #plt.colorbar()
        st.pyplot(fig)






    #st.pyplot(fig, width = "stretch")

if __name__ == "__main__":
    main()
