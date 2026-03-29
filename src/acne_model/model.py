#for training of actual model
#imports

import pandas as pd
import numpy as np
from collections import defaultdict
from functools import reduce
import statsmodels.api as sm
import json
from scipy.stats import invwishart, multivariate_normal
from scipy.linalg import eigvals, solve_continuous_lyapunov
from scipy.spatial import distance_matrix
from scipy.stats import entropy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from functools import partial

def old_unused_fit_predictive_linear_regression_model_of_severity(metadata_dfs):
    """Function to fit a predictive linear model of acne severity as a function of:
    1) Lagged/previous day's severity; 2/3) Cumulative days of the current treatment (in this case, either antibiotics or cream)
    4) Synergistic effect of cream being followed by a certain number of days of antibiotics.
    5) Saturation function based on the Michaelis-Menten half saturation constant (particularly as cream eventually causes the molecular system
    of skin cells to approach homeostasis, with its effect progressively leveling off."""

    day_and_sev = defaultdict(list)
    cream_saturation_constant = 12  # this hyperparameter can be changed as the model learns, at this time it's about half of the cream treatment block

    for metadata_df in metadata_dfs:
        for i, row in metadata_df.iterrows():
            last_severity = 0  # fix this later to be average baseline
            current_history = tuple(row["treatment_history"])

            # 4 variables - days of current antibiotics, days of current cream, cumulative cream/antibiotic effect, saturation function
            # unpacking history to get the relevant pieces, saving as keys
            current_treatment = current_history[len(current_history) - 1][0]
            days_current_treatment = current_history[len(current_history) - 1][1]
            # finding days of antibiotics if following cream
            days_of_current_antibiotics = 0
            days_of_current_cream = 0
            antibiotics_cream_interaction = 0
            saturation_function_output = 0

            if current_treatment == "Antibiotics":  # at some point this can be changed away from hardcoding
                days_of_current_antibiotics = days_current_treatment
                if len(current_history) != 1:  # ie, not the first treatment in a series
                    last_treatment = current_history[len(current_history) - 2][0]
                    if last_treatment == "Cream":  # this part gives syngergistic effect of antibiotics used after cream
                        previous_cream_days = current_history[len(current_history) - 2][1]
                        # simple linear function of the days of antibiotics giving the antibiotics_cream_interaction
                        antibiotics_cream_interaction = previous_cream_days * days_of_current_antibiotics

            if current_treatment == "Cream":
                days_of_current_cream = days_current_treatment
                saturation_function_output = days_of_current_cream / (cream_saturation_constant + days_of_current_cream)

            current_severity = row["AcneSeverity"]
            model_contributions = (
            {"days_of_current_antibiotics": days_of_current_antibiotics, "days_of_current_cream": days_of_current_cream,
             "antibiotics_cream_interaction": antibiotics_cream_interaction,
             "saturation_function_output": saturation_function_output},
            last_severity, current_severity)

            day_and_sev[current_history].append(model_contributions)
            last_severity = current_severity

    # saving the independent variable values per history to a dataframe for easier fitting to the multivariate regression model
    rows = []
    for history, entries in day_and_sev.items():
        for features, previous_sev, curr_sev in entries:
            delta = curr_sev - previous_sev
            row = features.copy()
            row['delta_severity'] = delta
            row['treatment_history'] = str(history)
            rows.append(row)

    dataframe_for_fitting = pd.DataFrame(rows)
    history_vectors_of_independent_vars = dataframe_for_fitting[['days_of_current_antibiotics', 'days_of_current_cream',
                                                                 'antibiotics_cream_interaction',
                                                                 'saturation_function_output']]
    history_vectors_of_independent_vars = sm.add_constant(history_vectors_of_independent_vars)  # adding constant term
    severity_change = dataframe_for_fitting["delta_severity"]

    fitted_model = sm.OLS(severity_change, history_vectors_of_independent_vars).fit()
def layer_1_bml_observable_biomarkers(design_matrix, imputed_biomarkers_matrix, prior_mean_matrix, prior_precision_matrix, prior_scale_matrix, dof):
    """Solves for the multivariate normal distribution between imputed biomarker vectors and observed clinical biomarkers.
    Priors for covariance between components, between observed imputed biomarkers pairwise, and
    coefficient matrices can be specified, otherwise uniformative priors are used."""
    #mental note: need to decide if parsing of raw df into design and imputed biomarkers matrices should be done here or in the raw data analysis function
    #also need to check the math here to see if it is accurate especially the fact the prior B matrix isn't used

    #Moore-Penrose Pseudoinverse for initial inspection later
    #coefficient_matrix_estimate = np.linalg.pinv(design_matrix) @ imputed_biomarkers_matrix

    #Closed Form Likelihood Solution
    #using the typical Inverse Wishart Prior for error covariance matrix, and Normal prior for coefficient matrix
    #noise_norm_cov_matrix_prior = invwishart(df = prior_precision_matrix.ndim, scale = prior_precision_matrix) #need to do a literature search about scale matrix covariance
    #coefficient_matrix_prior = multivariate_normal(x = x, cov = np.kron(noise_norm_cov_matrix_prior, np.linalg.inv(prior_scale_matrix)))
    post_precision_matrix = (design_matrix.T @ design_matrix) + prior_precision_matrix
    post_precision_matrix_converted = post_precision_matrix.to_numpy()

    post_mean_matrix = np.linalg.inv(post_precision_matrix_converted) @ \
                       (design_matrix.T @ imputed_biomarkers_matrix + prior_precision_matrix @ prior_mean_matrix)
    post_mean_matrix_converted = post_mean_matrix.to_numpy()

    converted_design_matrix = design_matrix.to_numpy()
    converted_mean_matrix = post_mean_matrix.to_numpy()

    residuals = imputed_biomarkers_matrix.to_numpy() - (converted_design_matrix @ converted_mean_matrix)
    mean_diff = post_mean_matrix_converted - prior_mean_matrix
    #ensuring post scale matrix is PSD
    post_scale_matrix_converted = prior_scale_matrix + \
                        (residuals.T @ residuals) + \
                        (mean_diff.T @ prior_precision_matrix @ mean_diff)

    post_dof = dof + len(post_mean_matrix.T)

    first_layer_params = {"precision_matrix": post_precision_matrix_converted, "post_mean_matrix": post_mean_matrix_converted,
              "post_scale_matrix": post_scale_matrix_converted, "post_dof": post_dof, "design_matrix": design_matrix, "imputed_biomarkers_matrix": imputed_biomarkers_matrix}

    return first_layer_params

def maximization_compute_cross_covs(latent_states, latent_state_covs, latent_states_lagged_covs):
    #computing state correlations and serial lag correlations at the same time
    state_correlations = np.zeros((latent_states[0].shape[0], latent_states[0].shape[0]))
    one_lag_correlations = np.zeros((latent_states[0].shape[0], latent_states[0].shape[0]))
    prev_state_correlations = np.zeros((latent_states[0].shape[0], latent_states[0].shape[0]))

    last_latent_state = latent_states[0]
    for i, (latent_state, smoothed_cov, latent_state_lagged_cov) in enumerate(zip(latent_states, latent_state_covs, latent_states_lagged_covs)):
        one_state_correlation_mean = latent_state @ latent_state.T + smoothed_cov
        one_serial_lag_correlation_mean = latent_state @ last_latent_state.T + latent_state_lagged_cov

        state_correlations += one_state_correlation_mean
        one_lag_correlations += one_serial_lag_correlation_mean

        if i > 1:
            prev_state_correlations += one_state_correlation_mean

    cross_covs = {"state_correlations": state_correlations, "one_lag_correlations": one_lag_correlations, "prev_state_correlations": prev_state_correlations}

    return cross_covs

def compute_cross_layer_moments(layer_2_mapping, all_within_layers_moments, first_layer_params):
    """Also solves for the latent state serial covariance, Cov(v_t, v_t-1), using OU process params."""
    here_1st_precision_matrix = first_layer_params["precision_matrix"]
    here_2nd_precision_matrix = np.linalg.inv(all_within_layers_moments["2nd_layer_posterior_covariance"])

    joint_covariance_z_y = np.array([here_1st_precision_matrix + layer_2_mapping.T @ here_2nd_precision_matrix @ layer_2_mapping, -layer_2_mapping.T @ here_2nd_precision_matrix],
                                [-here_2nd_precision_matrix @ layer_2_mapping, here_2nd_precision_matrix])
    cross_layer_moment = joint_covariance_z_y + all_within_layers_moments["latent_state_forward_estimate"] @ all_within_layers_moments["posterior_imputed_mean"].T

    cross_layer_lagged_moments = {"joint_covariance_z_y": joint_covariance_z_y, "cross_layer_moment": cross_layer_moment }

    return cross_layer_lagged_moments

def layer_2_imputed_biomarkers_multivar_model(first_layer_params, this_last_observables_cov, this_last_mean, this_last_coeff_matrix, ou_matrices, current_ls_ev_F, current_imp_map_H, last_initial_state, current_process_noise_cov, current_measurement_noise_cov):
    """Fits layer 2 of the imputed biomarker multivariate model. Implements one iteration of EM."""
    # ends with learning the relative contributions of the imputed biomarkers to total bacterial dysbiosis, total sebum production, total inflammation

    usable_imputeds_matrix = first_layer_params["imputed_biomarkers_matrix"].to_numpy()
    #---Expectation Step
    expectation_results = all_layers_expectation_step(last_observables_cov=this_last_observables_cov, last_mean=this_last_mean, last_coeff_matrix=this_last_coeff_matrix, first_layer_params=first_layer_params,
                                                     last_ls_ev_F=current_ls_ev_F, last_imp_map_H=current_imp_map_H, last_process_noise_cov=current_process_noise_cov, last_measurement_noise_cov=current_measurement_noise_cov,
                                                     imputed_biomarkers_matrix=usable_imputeds_matrix, last_initial_latent_state=last_initial_state, last_process_cov= this_last_observables_cov)

    unpacked_latent_states  = expectation_results["latent_states"]
    latent_state_covs = expectation_results["latent_state_covs"]
    latent_state_lagged_covs = expectation_results["lagged_latent_state_covs"]
    latent_state_resids = expectation_results["latent_state_resids"]


    #---Maximization Step
    this_maximization = full_maximization_step_HBM(latent_states = unpacked_latent_states,
                                                   smoothed_covariances = latent_state_covs, lagged_covariances = latent_state_lagged_covs, imputed_biomarkers=usable_imputeds_matrix)


    #---Computation of Ornstein Uhlenbeck RV, indexed by its continous Lyanpunov equation solution matrix
    #this_OU_State_Matrix_const = ou_matrices["State_Matrix_OU"]
    #this_OU_State_Matrix = this_OU_State_Matrix_const * np.identity(cov_matrix_maximized.shape[0]) #check dimensions for Wiener process and State Matrix

    #this_OU_Weiner_Matrix_const = ou_matrices["Wiener_Matrix_OU"]
    #this_OU_Weiner_Matrix = this_OU_Weiner_Matrix_const * np.identity(cov_matrix_maximized.shape[0])
    #mental note: make sure covariance matrix for second normal is initialized and also the ones for the OU processes above
    #this_OU_index_matrix, test_result = multivar_Ornstein_Uhlenbeck(state_matrix=this_OU_State_Matrix, diffusion_matrix=this_OU_Weiner_Matrix)


    #final_cov_matrix_maximized = cov_matrix_maximized + this_OU_index_matrix

    return "ok"

def backpropagate_KG_to_layer_1():
    pass
def all_layers_expectation_step(last_observables_cov, last_mean, last_coeff_matrix, imputed_biomarkers_matrix, first_layer_params, last_ls_ev_F, last_imp_map_H, last_initial_latent_state, last_process_cov,
                                last_process_noise_cov, last_measurement_noise_cov):
    #computing first and second moments for layer 2
    #conducts all k Kalman filter/smoothings for the latent biological state z_{t} = [B_t, I_t, S_t]
    """Follows this latent state evolution model: z_{t+1) = F(z_{t}) + B(u_{t}) + w_{k}; w_{k} ~ N(0, Q)
    and this imputed biomarker to latent state mapping: y_{t+1} = H(y_{t}) + v_{k};  v_{k} ~ N(0, R).
    """
    rng = np.random.default_rng()
    latent_states = [last_initial_latent_state]

    latent_state_covs = [last_process_cov]
    lagged_latent_state_covs = []
    latent_state_resids = []
    last_a_post_estimate = last_initial_latent_state
    last_a_post_cov_est = last_process_cov


    kalman_gains = []
    smoothing_gains = []


    for iteration in range(0, len(imputed_biomarkers_matrix)):
        #---Prediction Step
        a_priori_state = last_ls_ev_F @ last_a_post_estimate #+B_k @ U_k
        #random_noise_realization_process = rng.normal(loc=np.zeros((last_ls_ev_F @ last_a_post_estimate).shape), scale=last_process_noise_cov, size=(last_ls_ev_F @ last_a_post_estimate).shape)
        a_priori_cov = last_ls_ev_F @ last_a_post_cov_est @ last_ls_ev_F.T + last_process_noise_cov

        #---Update Step
        random_noise_realization = rng.normal(loc=np.zeros( (last_imp_map_H @ a_priori_state).shape), scale=last_measurement_noise_cov, size=(last_imp_map_H @ a_priori_state).shape)
        pred_imputed = last_imp_map_H @ a_priori_state + random_noise_realization

        innovation = imputed_biomarkers_matrix[iteration] - pred_imputed
        innovation_cov = last_imp_map_H @ a_priori_cov @ last_imp_map_H.T + last_measurement_noise_cov

        opt_Kal_gain = a_priori_cov @ last_imp_map_H.T @ innovation_cov
        kg_size = opt_Kal_gain.shape

        a_post_state  = a_priori_state + opt_Kal_gain @ innovation
        a_post_cov = (np.identity(kg_size[0]) - opt_Kal_gain @ last_imp_map_H) @ a_priori_cov
        a_post_residual = pred_imputed - last_imp_map_H @ a_post_state

        #computing smoothing gain explicitly
        if iteration > 0:
            smoothing_gain =  last_a_post_cov_est @ last_ls_ev_F.T @ np.linalg.pinv(a_priori_cov) #pinverse for now, need to fix singularity
            smoothing_gains.append(smoothing_gain)



        latent_states.append(a_post_state)
        latent_state_covs.append(a_post_cov)
        latent_state_resids.append(a_post_residual)
        kalman_gains.append(opt_Kal_gain)

        #updating
        last_a_post_estimate = a_post_state
        last_a_post_cov_est = a_post_cov

    #calculating the lagged-1 covariances recursively
    current_future_smoothing = list(reversed(smoothing_gains))[0] #future one is seen as the last one, past one is seen as the future one
    #current_future_lag_cov = list(reversed(latent_state_covs))[0]
    for i, (smoothing_gain, a_post_cov) in enumerate(zip(reversed(smoothing_gains), reversed(latent_state_covs))):
        if  1 < i < len(smoothing_gains) - 1:
            current_past_lag_cov = latent_state_covs[i-1]
            lag_1_cov = a_post_cov @ current_future_smoothing.T + smoothing_gain @ (current_past_lag_cov - last_ls_ev_F @ a_post_cov) @ current_future_smoothing.T
            lagged_latent_state_covs.append(lag_1_cov)

    lagged_latent_state_covs.reverse()


    results = {"latent_states": latent_states, "latent_state_covs": latent_state_covs, "latent_state_resids": latent_state_resids, "kalman_gains": kalman_gains, "smoothing_gains": smoothing_gains,
               "lagged_latent_state_covs": lagged_latent_state_covs}

    return results
def full_maximization_step_HBM(latent_states, smoothed_covariances, lagged_covariances, imputed_biomarkers):
    """Maximizes in closed form."""
    sufficient_stats = maximization_compute_cross_covs(latent_states = latent_states,
                                                       latent_state_covs = smoothed_covariances, latent_states_lagged_covs = lagged_covariances)
    new_ls_ev_F = sufficient_stats["one_lag_correlations"] @ np.linalg.pinv(sufficient_stats["prev_state_correlations"])
    outer_products = [np.outer(imp, lat) for lat, imp in zip(latent_states, imputed_biomarkers)]
    sum_outer = np.sum(outer_products, axis=0)
    new_imp_map_H = sum_outer @ np.linalg.pinv(sufficient_stats["state_correlations"])

    new_process_cov_Q = 1/len(latent_states) * (sufficient_stats["state_correlations"] - new_ls_ev_F @ sufficient_stats["one_lag_correlations"].T)

    new_measurement_cov_R = np.zeros(new_imp_map_H.shape)

    for latent_state, imp_vector, smoothed_covariance in zip(latent_states, imputed_biomarkers, smoothed_covariances):
        one_term = (imp_vector - new_imp_map_H @ latent_state) @ (imp_vector - new_imp_map_H @ latent_state).T + new_imp_map_H @ smoothed_covariance @ new_imp_map_H.T
        new_measurement_cov_R += one_term

    #mental note: just need to close the loop of these new mappings and add backprogagation of Kalman gain into the layer 1 mapping
    #introduce NN to replace linearization next?
    return sufficient_stats

def layer_3_latent_state_severity_mapping(maximized_mean):
    mapping = 0
    #need to use the normal posterior mean from layer 2 as the likelihood for the gamma

def univariate_gaussian(mean, std_dev, dof):
    """Implementation of Gaussian function for below use."""
    coefficient = 1 / (std_dev * np.sqrt(2 * np.pi))
    exponent = -((dof - mean) ** 2) / (2 * std_dev ** 2)
    return coefficient * np.exp(exponent)

def multivar_Ornstein_Uhlenbeck(state_matrix, diffusion_matrix):
    """Uses the solution to the Fokker-Plank equation  to approximate the transition probability of realization of the OU process
    asserted to comprise remainder of latent state. Is multivariate.
    Returns the solution to the Lyanpunov equation indexing the given solution."""

    #---Checking if input is a numpy array and is square
    used_state_matrix = np.asarray(state_matrix)
    used_diff_matrix = np.asarray(diffusion_matrix)
    if used_state_matrix.shape[0] != used_state_matrix.shape[1]:
        raise ValueError("State Matrix must be square.")

    #---Checking Hurwitz Condition (negative eigenvals)
    state_eigenvals = eigvals(used_state_matrix)
    if not np.all(state_eigenvals.real < 0):
        return False, "Matrix is not Hurwitz (unstable)."

    try:
        q_term = used_diff_matrix @ used_diff_matrix.T
        lyanpunov_soln = solve_continuous_lyapunov(a = used_state_matrix, q = -q_term)
        return True, lyanpunov_soln
    except Exception as e:
        return False, str(e)

def process_dataframes_for_model(dataframes, raw_frames, K_d_insulin = 4.0, K_d_IGF1 = 1.0, gaussian_weights = [1/3, 1/3, 1/3],
                                 gaussian_params = [[4.0, 0.5], [5.5, 0.5], [7.6, 0.5]], control_rows = 28):
    """Function that processes dataframes into matrices in the right format for EM passage. Relies on formatting
    used in the raw_data_analysis module.
    Also finds the rows in terms of these mechanistic features/regression basis functions:
    {Insulin Conc./Insulin Conc. + K_d, IGF1 Conc/IGF1 Conc. + K_d, Gaussian Mixture Model(pH), NLR}.
    K_ds are in nM and are rough estimates from literature (kinetic and PK-PD modeling needed later).
    """
    #collecting observable and imputed biomarkers from each dataframe
    observables_names = ['Insulin Conc.','IGF1 Conc.', 'Skin pH', 'Bulk Androgen Conc.', 'NLR']
    imputed_names = ['mTORC1 Conc.', 'Dysbiosis', 'LKB4 Conc.']
    acne_severity = ["AcneSeverity"]
    observables_subframes_separate = [df[observables_names].reset_index(drop=True) for df in dataframes]
    imputeds_subframes_separate = [df[imputed_names].reset_index(drop=True) for df in dataframes]

    all_relevant_names = observables_names + imputed_names + acne_severity
    #baseline averages are collected here for later initialization
    control_rows_frames = [raw_df[all_relevant_names].head(control_rows) for raw_df in raw_frames]
    control_rows_tensor = []
    for relevant_column_name in all_relevant_names:
        all_columns = np.array([control_frame[relevant_column_name].to_numpy() for control_frame in control_rows_frames])
        single_matrix = all_columns.T
        control_rows_tensor.append(single_matrix)


    severities_subframes_separate = [df[["AcneSeverity"]].reset_index(drop=True) for df in dataframes]
    severities_full_frame_sum = reduce(lambda x, y: x.add(y, fill_value=0), severities_subframes_separate)
    severities_av_matrix = severities_full_frame_sum/len(severities_subframes_separate) #different approach than data driven version

    #simple average of all observations
    observables_full_frame_sum = reduce(lambda x, y: x.add(y, fill_value=0), observables_subframes_separate)
    raw_design_matrix = observables_full_frame_sum/len(observables_subframes_separate)
    #embedding raw design matrix rows into the mechanistic feature space
    raw_design_matrix["Insulin Occ."] = raw_design_matrix["Insulin Conc."] / (raw_design_matrix["Insulin Conc."] + K_d_insulin)
    raw_design_matrix["IGF1 Occ."] = raw_design_matrix["IGF1 Conc."] / (raw_design_matrix["IGF1 Conc."] + K_d_IGF1)
    androgen_feedback_num = raw_design_matrix["IGF1 Conc."] * raw_design_matrix["Bulk Androgen Conc."]
    androgen_feedback_denom = raw_design_matrix["IGF1 Conc."] + K_d_IGF1
    raw_design_matrix["IGF1-Androgen Feedback"]  = androgen_feedback_num/androgen_feedback_denom

    #outputting result of GMM on pH here. Uses hyperparams that can be configured by the user.
    gaussian_evaluated = [gaussian_weights[i]*univariate_gaussian(mean=this_mean, std_dev=this_std, dof=raw_design_matrix["Skin pH"]) for i, (this_std, this_mean) in enumerate(gaussian_params)]
    gaussian_output = np.sum(gaussian_evaluated, axis = 0)
    raw_design_matrix["pH GMM"] = gaussian_output

    drop_columns = ["Insulin Conc.", "IGF1 Conc.", "Bulk Androgen Conc.", "Skin pH"]
    final_raw_design_matrix = raw_design_matrix.drop(columns = drop_columns)

    imputeds_full_frame_sum = reduce(lambda x, y: x.add(y, fill_value=0), imputeds_subframes_separate)
    imputeds_biomarkers_matrix = imputeds_full_frame_sum / len(imputeds_subframes_separate)

    return severities_av_matrix, final_raw_design_matrix, imputeds_biomarkers_matrix, control_rows_tensor, all_relevant_names

def control_diagnostics(control_tensor, dataframe_names):
    """Used to assess the distribution of baselines to check for steady state. Computes means, variances, and KL divergence betewen adjacent distributions."""
    control_statistics_matrices = []
    names = ["Means", "Variances", "KL_Divergence"]

    for single_matrix, name in zip(control_tensor, dataframe_names):
        means = []
        variances = []
        klds = []
        last_raw_distribution, last_bin_edges = np.histogram(single_matrix[0], density=True)
        last_true_distribution = last_raw_distribution * np.diff(last_bin_edges)

        for row in single_matrix:
            means.append(np.mean(row))
            variances.append(np.var(row))
            raw_distribution, bin_edges = np.histogram(row, density=True)
            true_distribution = raw_distribution * np.diff(bin_edges)
            klds.append(entropy(last_true_distribution, true_distribution))
            last_true_distribution = true_distribution
        control_statistics_matrix = np.vstack((means, variances, klds))
        control_stats_df = pd.DataFrame(control_statistics_matrix.T, columns=names)
        control_statistics_matrices.append(control_stats_df)

    #displaying diagnostic plots
    for outer_name, control_statistics_matrix in zip(dataframe_names, control_statistics_matrices):
        pass
        #means_plot = control_statistics_matrix.plot(kind = "line", x = None, y = "Means", title = f"Means {outer_name}")
        #vars_plot = control_statistics_matrix.plot(kind = "line", x = None, y = "Variances", title = f"Vars {outer_name}")
        #klds_plot = control_statistics_matrix.plot(kind = "line", x = None, y = "KL_Divergence", title = f"KLDs {outer_name}")

    #plt.show()

    control_statistics_as_dict = dict(zip(dataframe_names, control_statistics_matrices))

    return control_statistics_as_dict

def fit_HBM_model(initial_severities_matrix, initial_design_matrix, initial_biomarkers_matrix,
                  prior_mean_matrix, prior_precision_matrix, prior_scale_matrix, initial_2nd_layer_cov_mat, initial_2nd_layer_mean_mat,
                  initial_2nd_layer_coeff_mat, control_matrices, last_OU_matrices, initial_F, initial_H, initial_Q, initial_R,
                  dof = 0, max_iterations = 20):

    #Initialization of all model constants already completed in model config

    ##----Initializing the EM Loop
    last_design_matrix = initial_design_matrix
    last_biomarkers_matrix = initial_biomarkers_matrix
    last_severities_matrix = initial_severities_matrix

    last_2nd_layer_covariance_matrix = initial_2nd_layer_cov_mat
    last_2nd_layer_mean_matrix = initial_2nd_layer_mean_mat
    last_2nd_layer_coeff_matrix = initial_2nd_layer_coeff_mat
    processed_last_OU_matrices = {"State_Matrix_OU": last_OU_matrices["State_Matrix_OU"] * np.identity(last_biomarkers_matrix.shape[1]),
                                  "Wiener_Matrix_OU": last_OU_matrices["Wiener_Matrix_OU"] * np.identity(last_biomarkers_matrix.shape[1])}

    #['mTORC1 Conc.', 'Dysbiosis', 'LKB4 Conc.']
    imputeds_list = [control_matrices['mTORC1 Conc.']["Means"].to_numpy(),
                     control_matrices['Dysbiosis']["Means"].to_numpy(),
                     control_matrices["LKB4 Conc."]["Means"].to_numpy()]

    imputeds_mean = np.array([np.mean(control_matrices['mTORC1 Conc.']["Means"].to_numpy()), np.mean(control_matrices['Dysbiosis']["Means"].to_numpy()), np.mean(control_matrices["LKB4 Conc."]["Means"].to_numpy())])



    this_initial_latent_state = np.linalg.inv(initial_H @ initial_H.T) @ initial_H @ imputeds_mean
    this_initial_covariance = np.cov(m = imputeds_list, rowvar=True)
    #print(this_initial_latent_state, this_initial_covariance)

    for em_iteration in range(max_iterations):
        one_iter_layer1_params = layer_1_bml_observable_biomarkers(design_matrix=last_design_matrix, imputed_biomarkers_matrix=last_biomarkers_matrix,
                                                                 prior_mean_matrix=prior_mean_matrix, prior_precision_matrix=prior_precision_matrix, prior_scale_matrix=prior_scale_matrix, dof = dof)
        one_iter_layer2_params = layer_2_imputed_biomarkers_multivar_model(first_layer_params=one_iter_layer1_params, this_last_observables_cov=last_2nd_layer_covariance_matrix, ou_matrices=processed_last_OU_matrices,
                                                                           this_last_mean=last_2nd_layer_mean_matrix, this_last_coeff_matrix=last_2nd_layer_coeff_matrix, current_ls_ev_F=initial_F, current_imp_map_H=initial_H,
                                                                           current_measurement_noise_cov=initial_Q, current_process_noise_cov=initial_R, last_initial_state=this_initial_latent_state)


def process_data_and_build_HBM(these_dataframes, model_priors_and_config_handle, these_raw_frames):
    this_ism, this_ids, this_ibm, this_control_tensor, dataframe_column_names_ordered = process_dataframes_for_model(these_dataframes, these_raw_frames)
    this_control_statistics_matrices = control_diagnostics(this_control_tensor, dataframe_column_names_ordered)

    with open(model_priors_and_config_handle, "r") as config:
        model_priors_and_config = json.load(config)


    this_pmm = model_priors_and_config["Prior_Mean"] * np.ones((this_ids.shape[1], this_ibm.shape[1])) #should be shape 5,3
    this_ppm = model_priors_and_config["Prior_Precision"] * np.identity(this_ids.shape[1])
    this_pcm = model_priors_and_config["Prior_Scale"] * np.identity(this_ibm.shape[1])

    prior_dof = model_priors_and_config["DOF"] + 1
    this_2nd_cov = this_pcm / (prior_dof - this_ibm.shape[0] - 1)
    this_2nd_mean = model_priors_and_config["Prior_2nd_Mean"] * np.ones((this_ibm.shape[1], 1)) * model_priors_and_config["Prior_2nd_Mean"]
    this_2nd_coeff_matrix = model_priors_and_config["Prior_2nd_Coeff_Matrix"] * np.identity(this_ibm.shape[1])

    initial_ls_ev_f = model_priors_and_config["Initial_LS_EV_F"] * np.identity(this_pmm.shape[1])
    initial_imp_map_H = np.diag(model_priors_and_config["Initial_IMP_Map_H"] * np.ones(this_pmm.shape[1]))
    initial_q_process_noise = model_priors_and_config["Initial_Q_Process_Noise"] * np.ones(this_pmm.shape[1]) #changed
    initial_r_measurement_noise = model_priors_and_config["Initial_R_Measurement Noise"] * np.ones(this_2nd_mean.shape[0])

    last_OU_matrices = {"State_Matrix_OU": model_priors_and_config["State_Matrix_OU"], "Wiener_Matrix_OU": model_priors_and_config["Wiener_Matrix_OU"]}  # placeholder

    this_fit_model = fit_HBM_model(initial_severities_matrix = this_ism, initial_design_matrix=this_ids, initial_biomarkers_matrix=this_ibm,
                                    prior_mean_matrix=this_pmm, prior_precision_matrix=this_ppm, prior_scale_matrix=this_pcm,
                                    dof = prior_dof, initial_2nd_layer_cov_mat=this_2nd_cov, initial_2nd_layer_mean_mat= this_2nd_mean,
                                    initial_2nd_layer_coeff_mat=this_2nd_coeff_matrix, last_OU_matrices=last_OU_matrices,
                                    initial_F=initial_ls_ev_f, initial_H=initial_imp_map_H, initial_Q=initial_q_process_noise, initial_R=initial_r_measurement_noise, control_matrices=this_control_statistics_matrices)



def reparameterize(model_params):
    """Used to reparameterize most existing model parameters to improve identifiability."""

    I_decay_total =  model_params[
        "I_baseline_decay"]  # total inflammatory decay due to decay (not actually reparameterized without tstd)
    I_drive_bacterial_relative = model_params["I_bacterial_induction"] * model_params[
        "r_growth"]  # bacterial inflammation induction with growth rate complementing
    I_drive_bacterial_ratio = model_params["I_bacterial_induction"] / model_params[
        "r_growth"]  # bacterial inf induction with respect to growth rate

    t = 1 / model_params["r_growth"]  # period of bacterial growth per individual (common timescale)
    # normalizing all rate constants by the timescale
    k_sebum_scaled = model_params["k_sebum"] * t
    k_antibiotics_scaled = model_params["k_antibiotics"] * t
    I_decay_scaled = I_decay_total * t

    reparameterized_dict = {"I_decay_scaled": I_decay_scaled,
                            "I_drive_bacterial_relative": I_drive_bacterial_relative,
                            "I_drive_bacterial_ratio": I_drive_bacterial_ratio,
                            "k_sebum_scaled": k_sebum_scaled, "k_antibiotics_scaled": k_antibiotics_scaled, "t": t,
                            "r_cream_clean": model_params["r_cream_clean"],
                            "r_I_production": model_params["r_I_production"], "K_CC": model_params["K_CC"]}
    return reparameterized_dict

def state_evolution_vv(v_t_last, params, history, raw_distribution, severity_deltas, index = 0):
    """Explicit function for the state evolution function, F_theta.
    Computes the next latent state, tstd.
    Then, it returns them along with the days of antibiotics used up to treatment day t and either 1 or 0 if cream was used on day t.  """
    prev_bac, prev_inf, prev_sebum = v_t_last
    # changed for testing purposes
    distribution_alphas = np.array(raw_distribution) if raw_distribution is not None else None
    probs = distribution_alphas / np.sum(distribution_alphas) if distribution_alphas is not None else None
    # ensuring severity deltas is a 1D numeric array
    severity_array = np.array(list(severity_deltas.values()) if isinstance(severity_deltas, dict) else severity_deltas,
                              dtype=float)
    #removed tstd from evolution to infer from clustering analysis below
    #frozen_tstd = 0.1
    #tstd_term = np.sum(probs * severity_array) if probs is not None else frozen_tstd + .05 * index
    tstd_term = 0

    days_antib = history[-1][1] if history[-1][0] == "Antibiotics" else 0
    was_cream_used = 1 if history[-1][0] == "Cream" else 0

    # reparameterization below
    # unpacking for readability
    t = params["t"]
    K_CC = params["K_CC"]

    # scaled rates
    k_abx = params["k_antibiotics_scaled"] / t
    k_seb = params["k_sebum_scaled"] / t

    # inflammation
    #f = params["I_decay_fraction_tstd"]
    I_decay_total = params["I_decay_scaled"] / t

    I_baseline_decay = I_decay_total
    #removed point estimate tstd
    #I_decay_tstd = (f / (1 + f)) * I_decay_total

    # bacterial inflammation drive
    I_bacterial_induction = np.sqrt(
        params["I_drive_bacterial_relative"] *
        params["I_drive_bacterial_ratio"]
    )

    # state vector components
    cur_bac = (
            prev_bac
            + (1 / t) * prev_bac * ((1 - prev_bac) / K_CC)
            - k_abx * days_antib * prev_bac
            + k_seb * prev_bac * prev_sebum
    )
    #removed point estimate tstd
    cur_inf = (
            prev_inf
            + I_bacterial_induction * prev_bac
            - I_baseline_decay * prev_inf
    )

    cur_sebum = (
            prev_sebum
            + params["r_I_production"] * prev_inf
            - params["r_cream_clean"] * was_cream_used
    )

    # clipping each returned value to avoid computational issues
    cur_bac = np.clip(cur_bac, 0, 10)
    cur_inf = np.clip(cur_inf, 0, 10)
    cur_sebum = np.clip(cur_sebum, 0, 10)

    return np.array([cur_bac, cur_inf, cur_sebum]), tstd_term, days_antib, was_cream_used


def log_prior_reparam(params, priors={
    # timescale (positive, log-normal)
    "t": (0.0, 0.5),

    # scaled rates
    "k_sebum_scaled": (0.0, 0.5),
    "k_antibiotics_scaled": (0.0, 0.5),

    # inflammation decay structure
    "I_decay_scaled": (0.0, 0.5),

    # bacterial → inflammation coupling
    "I_drive_bacterial_relative": (0.0, 0.5),
    "I_drive_bacterial_ratio": (0.0, 0.5),

    # other params
    "r_I_production": (np.log(0.035), 0.5),
    "r_cream_clean": (np.log(0.125), 0.5),

    # carrying capacity
    "K_CC": (np.log(1.0), 0.5),
}):
    """
    Log-prior modified to work entirely in reparameterized space.
    Assumes all parameters are positive and uses log-normal penalties.
    """
    lp = 0.0
    for key, (mu, sigma) in priors.items():
        if key in params:
            x = params[key]
            # guard against numerical death
            if x <= 0:
                return -np.inf
            lp += -0.5 * ((np.log(x) - mu) / sigma) ** 2
    return lp

def compute_log_likelihood(raw_distributions, pred_state_vectors, scoring_params, model_parameters, severity_deltas, Q=np.diag([1e-1, 1e-1, 1e-1])):
    """This function is wrapped by optimize_parameters. It uses maximum-likelihood estimation
    to estimate the variance in the residuals between empirical distributions of acne severity state and distributions of
    acne severity state conditioned on a latent state.
    Now also includes transition term to include a generative prior over latent states to improve identifiability,
    based on a small covariance matrix Q, its inverse, and its determinant.
    """

    total_log_l = 0.0
    Q_inverse = np.linalg.inv(Q) if Q is not None else None
    log_det_Q = np.log(np.linalg.det(Q)) if Q is not None else 0.0

    for i, (history, raw_distribution) in enumerate(raw_distributions.items()):
        predicted_vector_this_history = pred_state_vectors[history]
        predicted_probs = map_latent_states_to_probs(predicted_vector_this_history, scoring_params)

        # avoid log(0)
        predicted_probs = np.clip(predicted_probs, 1e-10, 1.0)

        distribution_alphas = np.array(raw_distribution)
        dirichlet_probs = distribution_alphas / np.sum(distribution_alphas)

        # Expected log-likelihood contribution for this history
        logL_history = np.sum(dirichlet_probs * np.log(predicted_probs))
        total_log_l += logL_history

        # now calculating transition likelihood with previous predicted latent state vector
        if i > 0:
            last_history = list(raw_distributions.keys())[i - 1]
            last_raw_distribution = list(raw_distributions.values())[i - 1]
            v_prev = pred_state_vectors[last_history]
            v_prediction = \
            state_evolution_vv(v_prev, model_parameters, last_history, last_raw_distribution, severity_deltas)[0]
            difference = predicted_vector_this_history - v_prediction
            transition_ll = -0.5 * difference.T @ Q_inverse @ difference - 0.5 * log_det_Q
            total_log_l += transition_ll

    total_log_l += log_prior_reparam(model_parameters)

    return total_log_l


def compute_log_likelihood_trans_matrix(empirical_counts, empirical_transition_matrices, latent_states, scoring_hyperparams, which_row = 0):
    """Computes log-likelihood for one transition-matrix row i under a multinomial model with log-offset."""
    W = scoring_hyperparams["scoring"]
    b = scoring_hyperparams["biases"]
    total_ll = 0.0

    for P_hat, C, z in zip(list(empirical_transition_matrices.values()), list(empirical_counts.values()), list(latent_states.values())):
        log_offset = np.log(P_hat[which_row] + 1e-12)
        counts = C[which_row]

        logits = log_offset + W @ z + b
        logits -= np.max(logits)
        p = np.exp(logits)
        p /= np.sum(p)

        total_ll += np.sum(counts * np.log(p + 1e-12))

    return total_ll

def softmax(score):
    """Standalone softmax function."""
    exp_score = np.exp(score - np.max(score))
    return exp_score / np.sum(exp_score)

def map_latent_states_to_probs(latent_state, scoring_hyperparameters, additional_term = 0):
    """Function used in the expectation step. It maps a latent state to a distribution over the 3 discrete acne severity change states.
    The distribution is produced with a weighted sum (score) of how coupled each severity state is to each of the components of the latent state.
    Softmaxing converts scores into probabilities.
    Contains additional term option for log-offset (used in transition matrix gradient descent)."""

    def softmax(x):
        # subtract max for numerical stability
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    # Get weights and biases
    W = np.array(scoring_hyperparameters["scoring"], dtype=float)  # shape (3,3)
    b = np.array(scoring_hyperparameters["biases"], dtype=float)  # shape (3,)

    # Normalize latent state to prevent huge values??
    #latent_norm = latent_state / (np.linalg.norm(latent_state) + 1e-6)

    # Linear scoring
    score = W @ latent_state + b + additional_term

    # Clip scores before softmax
    score = np.clip(score, -10, 10)

    # Softmax to probabilities
    probabilities = softmax(score)

    return probabilities
def unpack_reparameterized_params(reparam):
    """Used to return reparamterized parameters  """
    params = {}

    # Bacterial growth period/timescale
    t = reparam["t"]
    r_growth = 1.0 / t
    params["r_growth"] = r_growth

    # Bacterial inflammation induction
    rel = reparam["I_drive_bacterial_relative"]
    ratio = reparam["I_drive_bacterial_ratio"]

    I_bacterial_induction = np.sqrt(rel * ratio)
    params["I_bacterial_induction"] = I_bacterial_induction

    # Inflammatory decay
    I_decay_scaled = reparam["I_decay_scaled"]
    I_decay_total = I_decay_scaled / t

    #removed point estimate tstd

    I_baseline_decay = I_decay_total /1.0

    params["I_baseline_decay"] = I_baseline_decay

    # Interaction rates
    params["k_sebum"] = reparam["k_sebum_scaled"] / t
    params["k_antibiotics"] = reparam["k_antibiotics_scaled"] / t

    # Unaltered parameters
    params["r_cream_clean"] = reparam["r_cream_clean"]
    params["r_I_production"] = reparam["r_I_production"]
    params["K_CC"] = reparam["K_CC"]

    return params
def smm_model(prev_state, params, raw_distributions, severity_deltas, index, Q=np.diag([1e-1, 1e-1, 1e-1])):
    """Wrapper for state_evolution_vv. Uses state_evolution_vv to evaluate the model's Jacobian at the previous state
    for propagation of state uncertainity in Kalman Filtering.
    Returns that output along with the outputs of state_evolution_vv at the previous state."""
    history, raw_distribution = list(raw_distributions.items())[index]

    # Compute evolution (next latent state)
    evolution_output, tstd_term, days_antib, was_cream_used = state_evolution_vv(prev_state, params, history,
                                                                                 raw_distribution, severity_deltas)
    # evolution output is the predicted next latent state
    stochastic_evolution_output = evolution_output + np.random.multivariate_normal(mean=np.zeros(3),
                                                                                   cov=Q)  # making predicted vectors random
    return stochastic_evolution_output, tstd_term, days_antib, was_cream_used

def optimize_hyperparameters(raw_distributions, predicted_states, scoring_hyperparams, model_params, severity_deltas,
                             learning_rate=1e-5, maximum_steps=30, clip_value=5.0):
    """Function that implements gradient ascent to maximize the log-likelihood of the model's hyperparameters
    used in map_latent_states_to_severity_probs."""
    lls = []  # log likelihoods
    # ensure float arrays
    scoring_hyperparams["scoring"] = np.array(scoring_hyperparams["scoring"], dtype=float)
    scoring_hyperparams["biases"] = np.array(scoring_hyperparams["biases"], dtype=float)

    def compute_gradient(raw_distributions, predicted_latent_states, parameters):

        """Inner function for computing gradient of log-likelihood function with respect to scoring weights and biases."""
        num_states = len(list(raw_distributions.values())[0])
        grad_W = np.array(scoring_hyperparams["scoring"], dtype=float)
        grad_b = np.array(scoring_hyperparams["biases"], dtype=float)

        for history, raw_alphas in raw_distributions.items():
            v_h = predicted_latent_states[history]

            P = map_latent_states_to_probs(v_h, parameters)  # softmax probs
            distribution_alphas = np.array(raw_alphas)
            empirical_probs = distribution_alphas / np.sum(distribution_alphas)
            grad_W += np.outer(empirical_probs - P, v_h)
            grad_b += (empirical_probs - P)

        # clipping gradients to prevent issues
        grad_W = np.clip(grad_W, -clip_value, clip_value)
        grad_b = np.clip(grad_b, -clip_value, clip_value)

        return {"scoring": grad_W, "biases": grad_b}

    # actual execution
    for iteration in range(maximum_steps):
        gradients = compute_gradient(raw_distributions, predicted_states, scoring_hyperparams)
        scoring_hyperparams["scoring"] += (float(learning_rate) * gradients["scoring"])
        scoring_hyperparams["biases"] += (float(learning_rate) * gradients["biases"])
        ll = compute_log_likelihood(raw_distributions, predicted_states, scoring_hyperparams, model_params,
                                    severity_deltas)
        lls.append(ll)

    return scoring_hyperparams, lls
def optimize_hyperparameters_transmatrix_row(transition_kernels, predicted_states, which_row, empirical_counts, scoring_hyperparams,
                             learning_rate=float(1e-5), maximum_steps=30, clip_value=5.0):
    """Function that implements gradient ascent to maximize the log-likelihood of the model's hyperparameters
    used for transitions."""
    lls = []  # log likelihoods

    W_transition_string = "scoring_trans_row {}".format(which_row) #fixed keys
    b_transition_string = "biases_trans_row {}".format(which_row) #fixed keys

    #print("one iteration", transition_kernels)

    def compute_gradient_transmatrix(transition_kernels, predicted_latent_states, which_row, empirical_counts):
        """Inner function for computing gradient of log-likelihood function with respect to scoring weights and biases."""

        W_transition = np.array(scoring_hyperparams[W_transition_string])
        b_transition = np.array(scoring_hyperparams[b_transition_string])

        grad_W_transition = np.zeros_like(W_transition).astype('float64')
        grad_b_transition = np.zeros_like(b_transition).astype('float64')


        for p_hat, emp_counts, latent_state in zip(list(transition_kernels.values()), list(empirical_counts.values()), predicted_latent_states.values()):
            log_offset_prob = np.log(p_hat[which_row] + 1e-12).astype(float)
            actual_counts = emp_counts[which_row].astype(float)
            N = float(np.sum(actual_counts))


            logits = log_offset_prob + W_transition @ latent_state + b_transition
            logits -= np.max(logits)
            pred_prob_hat = np.exp(logits)
            pred_prob_hat/= np.sum(pred_prob_hat) #normalization


            slope = actual_counts - N * pred_prob_hat #fix casting problem somehow...something having to do with grad W transition???

            grad_W_transition += np.outer(slope, latent_state)
            grad_b_transition += slope

        return {W_transition_string: grad_W_transition.astype('float64'), b_transition_string: grad_b_transition.astype('float64')} #fixed keys

    # actual execution
    for iteration in range(maximum_steps):
        gradients = compute_gradient_transmatrix(transition_kernels, predicted_states, which_row, empirical_counts)
        #print(scoring_hyperparams[W_transition_string], float(learning_rate) * gradients[W_transition_string].astype('float64'))
        scoring_hyperparams[W_transition_string] += (learning_rate * gradients[W_transition_string].astype('float64')) #fixed keys
        scoring_hyperparams[b_transition_string] += (learning_rate * gradients[b_transition_string].astype('float64')) #fixed keys
        ll = compute_log_likelihood_trans_matrix(empirical_counts, transition_kernels, predicted_states, scoring_hyperparams)
        lls.append(ll)

    return scoring_hyperparams, lls

def optimize_process_and_measurement_covariances(pred_state_vectors, model_params, scoring_hyperparams, raw_distributions, severity_deltas):
    """
    Optimize the process (Q) and measurement (R) noise covariances.
    pred_state_vectors: dict of {history_name: latent_state_vector} from the E-step
    model_parameters: dict of current model parameters
    raw_distributions: dict of observed Dirichlet distributions
    severity_deltas: dict of severity changes
    Uses partials to make call to state evolution function above more straightforward.
    """

    # creating a partial of smm_model that fixes params, raw_distributions, severity_deltas
    state_evolution_fn = partial(
        smm_model,
        params=model_params,
        raw_distributions=raw_distributions)

    # ---- Process covariance Q ----
    q_residuals = []
    for history_index, (history, v_t) in enumerate(pred_state_vectors.items()):
        evolution_output, _, _, _, _, _ = state_evolution_fn(prev_state=v_t, severity_deltas=severity_deltas,
                                                             index=history_index)
        q_residuals.append(v_t - evolution_output)

    q_residuals = np.stack(q_residuals)
    Q = np.cov(q_residuals, rowvar=False)

    # ---- Measurement covariance R ----
    r_residuals = []
    for history_index, (history, v_t) in enumerate(pred_state_vectors.items()):
        y_obs = np.array(raw_distributions[history])
        y_obs = y_obs / np.sum(y_obs)  # normalize to probabilities
        y_pred = map_latent_states_to_probs(v_t, scoring_hyperparams)
        r_residuals.append(y_obs - y_pred)

    r_residuals = np.stack(r_residuals)
    R = np.cov(r_residuals, rowvar=False)

    return Q, R

def optimize_a_linear_parameter_set(target_values, input_matrix, mu=0.0, sigma=1.0, lognormal=True):
    """Function that uses ordinary least squares to optimize a set of linear parameter from the model that correspond to one component.
    Now enforces positivity by imposing a log normal prior (unless configured otherwise) onto linear parameters."""
    # OLS estimate
    theta = np.linalg.pinv(input_matrix) @ target_values

    # enforce positivity
    theta = np.maximum(theta, 1e-8)

    if lognormal:
        log_theta = np.log(theta)
        penalty_grad = log_theta / sigma ** 2
        theta -= 0.01 * penalty_grad  # small step toward prior mode

    return theta

def adjust_empirical_kernel(scoring_hyperparams, latent_state, empirical_kernel, number_states = 3):
    """Function to adjust an empirical kernel as a function of latent state. Used in Streamlit implementation."""
    all_rows = []
    for row_index, row in enumerate(empirical_kernel):
        adjusted_row = softmax(np.log(row) + np.dot(scoring_hyperparams["scoring_trans_row {}".format(row_index)], latent_state) + scoring_hyperparams["biases_trans_row {}".format(row_index)])
        all_rows.append(adjusted_row)
    adjusted_kernel = np.array(all_rows).reshape((number_states, number_states))
    return adjusted_kernel


def full_maximimzation_step(pred_state_vectors, raw_distributions, parameters, model_params, prev_state_vectors,days_antibiotics,cream_useds,severity_deltas, empirical_counts, empirical_kernels, learning_rate=1e-4,max_grad_steps=30):
    #need to add optimization of Markov process Kernels
    # 1. Optimize nonlinear scoring weight and bias hyperparameters
    scoring_params, scoring_ll = optimize_hyperparameters(
        raw_distributions=raw_distributions,
        predicted_states=pred_state_vectors,
        scoring_hyperparams=parameters,
        model_params=model_params,
        severity_deltas=severity_deltas,
        learning_rate=learning_rate,
        maximum_steps=max_grad_steps)

    parameters["scoring"] = scoring_params["scoring"]
    parameters["biases"] = scoring_params["biases"]

    #calling optimization for each row of the matrices
    #row 1
    scoring_params_trans_matrices_row_1, trans_scoring_ll_1 = optimize_hyperparameters_transmatrix_row(
        transition_kernels=empirical_kernels, predicted_states=pred_state_vectors, which_row = 0,
        empirical_counts=empirical_counts, scoring_hyperparams=parameters)

    scoring_params_trans_matrices_row_2, trans_scoring_ll_2 = optimize_hyperparameters_transmatrix_row(
        transition_kernels=empirical_kernels, predicted_states=pred_state_vectors, which_row=1,
        empirical_counts=empirical_counts, scoring_hyperparams=parameters)

    scoring_params_trans_matrices_row_3, trans_scoring_ll_3 = optimize_hyperparameters_transmatrix_row(
        transition_kernels=empirical_kernels, predicted_states=pred_state_vectors, which_row=2,
        empirical_counts=empirical_counts, scoring_hyperparams=parameters)

    parameters["scoring_trans_row 0"] = scoring_params_trans_matrices_row_1["scoring_trans_row 0"]
    parameters["biases_trans_row 0"] = scoring_params_trans_matrices_row_1["biases_trans_row 0"]

    parameters["scoring_trans_row 1"] = scoring_params_trans_matrices_row_2["scoring_trans_row 1"]
    parameters["biases_trans_row 1"] = scoring_params_trans_matrices_row_2["biases_trans_row 1"]

    parameters["scoring_trans_row 2"] = scoring_params_trans_matrices_row_3["scoring_trans_row 2"]
    parameters["biases_trans_row 2"] = scoring_params_trans_matrices_row_3["biases_trans_row 2"]


    # Helper function to safely slice lists to match target length
    def safe_slice(lst, target_len):
        return lst[:target_len] if len(lst) > target_len else lst

    # 3a. Bacterial component
    bac_targets = np.array([v[0] for v in pred_state_vectors.values()])

    bac_inputs = np.column_stack([
        [v_prev[0] for v_prev in prev_state_vectors.values()],
        safe_slice(days_antibiotics, len(bac_targets)),
        [v_prev[2] for v_prev in prev_state_vectors.values()]])

    # Fit scaled parameters
    new_t, new_k_antib_scaled, new_k_sebum_scaled = optimize_a_linear_parameter_set(bac_targets, bac_inputs)
    model_params["t"] = np.clip(new_t, 1e-3, 10.0)
    model_params["k_antibiotics_scaled"] = np.clip(new_k_antib_scaled, 1e-6, 10.0)
    model_params["k_sebum_scaled"] = np.clip(new_k_sebum_scaled, 1e-6, 10.0)

    # 3b. Inflammation component
    inf_targets = np.array([v[1] for v in pred_state_vectors.values()])

    #point estimates tstds removed
    inf_inputs = np.column_stack([
        [v_prev[0] for v_prev in prev_state_vectors.values()],
        [v_prev[1] for v_prev in prev_state_vectors.values()]])

    new_I_drive_rel, new_I_decay_total = optimize_a_linear_parameter_set(
        inf_targets, inf_inputs)

    model_params["I_drive_bacterial_relative"] = np.clip(new_I_drive_rel, 1e-6, 10.0)
    model_params["I_decay_scaled"] = np.clip(new_I_decay_total * model_params["t"], 1e-6, 10.0)
    #model_params["I_decay_fraction_tstd"] = np.clip(new_I_decay_frac, 1e-3, 10.0)

    # 3c. Sebum component
    seb_targets = np.array([v[2] for v in pred_state_vectors.values()])
    seb_inputs = np.column_stack([
        [v_prev[1] for v_prev in prev_state_vectors.values()],
        safe_slice(cream_useds, len(seb_targets))])

    new_r_I_prod_scaled, new_r_cream_clean = optimize_a_linear_parameter_set(seb_targets, seb_inputs)
    model_params["r_I_production"] = new_r_I_prod_scaled
    model_params["r_cream_clean"] = new_r_cream_clean

    return parameters, model_params

def compute_observation_Jacobian_softmax(weights, biases, v_t):
    """Function that computes the Jacobian for Kalman gain, given the mapping between acne severity change state probability
    and latent state."""
    predicted_probs = weights @ v_t + biases
    predicted_probs_softmaxed = np.exp(predicted_probs - np.max(predicted_probs)) / np.sum(
        np.exp(predicted_probs - np.max(predicted_probs)))
    # Jacobian
    J_softmax = np.diag(predicted_probs_softmaxed) - np.outer(predicted_probs_softmaxed, predicted_probs_softmaxed)
    J_obs_softmax = J_softmax @ weights.T
    return J_obs_softmax

def compute_transition_log_likelihood(v_t, v_t_pred, Q):
    """Function used to add noise likelihood to existing likelihood (generative prior)."""
    diff = v_t - v_t_pred
    Q_inv = np.linalg.inv(Q)
    logdetQ = np.log(np.linalg.det(Q))
    return -0.5 * diff.T @ Q_inv @ diff - 0.5 * logdetQ

def fit_latent_state_space_model(empirical_counts, empirical_kernels, initial_parameters, raw_distributions, severity_deltas, model_config,
                                 max_iterations=20):
    """EM Filter fitting for acne latent state space model."""

    # --- Initial guesses for latent states and covariances---
    initial_states_guess = np.array([1.0, 1.0, 1.0])  # bacteria, inflammation, sebum


    # --- Last iteration values ---
    # scoring hyperparams for mapping of latent state to probability distribution
    # updating to now have 3 weights per acne severity change state instead of just vector of probabilities of each state
    updated_state_weights = np.array(model_config["scoring"])
    last_state_biases = np.array(model_config["biases"])
    last_state_weights_row_0 = np.array(model_config["scoring_trans_row 0"])
    last_state_biases_row_0 = np.array(model_config["biases_trans_row 0"])

    last_state_weights_row_1 = np.array(model_config["scoring_trans_row 1"])
    last_state_biases_row_1 = np.array(model_config["biases_trans_row 1"])

    last_state_weights_row_2 = np.array(model_config["scoring_trans_row 2"])
    last_state_biases_row_2 = np.array(model_config["biases_trans_row 2"])

    last_params = initial_parameters.copy()  # model parameters

    # --- Initialize for EM ---
    histories = list(raw_distributions.keys())
    first_history = histories[0]

    # Run smm_model for the first history
    z_first, t_first, days_first, cream_first = smm_model(initial_states_guess, last_params, raw_distributions,
                                                          severity_deltas, 0)
    # Store initial predictions
    v_predictions = {first_history: z_first}

    last_predictions = {h: initial_states_guess.copy() for h in histories}

    # ---- Full EM ----
    for em_iteration in range(max_iterations):
        t_tstds = []
        days_antibiotics = []
        cream_useds = []
        v_predictions = {h: last_predictions[h].copy() for h in histories}
        for history_index, current_history in enumerate(histories):
            # -----  Expectation Step
            current_history = histories[history_index]
            # current state's values (treatment day t) are found here
            z_current_prediction, predicted_tstd_term, here_days_antib, here_was_cream_used = smm_model(
                last_predictions[current_history], last_params, raw_distributions, severity_deltas, history_index)

            v_predictions[current_history] = z_current_prediction
            last_predictions[current_history] = z_current_prediction  # updating prev_state_vectors

            t_tstds.append(predicted_tstd_term)
            days_antibiotics.append(here_days_antib)
            cream_useds.append(here_was_cream_used)

        # ---- Maximization  ----
        # Create a partial of smm_model for Q/R optimization

        updated_parameters, last_params = full_maximimzation_step(
            pred_state_vectors=v_predictions,
            raw_distributions=raw_distributions,
            parameters={"scoring": updated_state_weights, "biases": last_state_biases,
                        "scoring_trans_row 0": last_state_weights_row_0, "biases_trans_row 0": last_state_biases_row_0,
                        "scoring_trans_row 1": last_state_weights_row_1, "biases_trans_row 1": last_state_biases_row_1,
                        "scoring_trans_row 2": last_state_weights_row_0, "biases_trans_row 2": last_state_biases_row_2},
            model_params=last_params,
            prev_state_vectors=last_predictions,
            days_antibiotics=days_antibiotics,
            cream_useds=cream_useds,
            severity_deltas=severity_deltas,
            empirical_counts=empirical_counts,
            empirical_kernels=empirical_kernels,
            learning_rate=1e-3,
            max_grad_steps=30
        )

        # now explicitly get the scoring vector
        updated_state_weights = np.array(updated_parameters["scoring"], dtype=float)
        updated_state_biases = np.array(updated_parameters["biases"], dtype=float)

        updated_row_0_weights = np.array(updated_parameters["scoring_trans_row 0"], dtype=float)
        updated_row_0_biases = np.array(updated_parameters["biases_trans_row 0"], dtype=float)

        updated_row_1_weights = np.array(updated_parameters["scoring_trans_row 1"], dtype=float)
        updated_row_1_biases = np.array(updated_parameters["biases_trans_row 1"], dtype=float)

        updated_row_2_weights = np.array(updated_parameters["scoring_trans_row 2"], dtype=float)
        updated_row_2_biases = np.array(updated_parameters["biases_trans_row 2"], dtype=float)



        #Updating scoring for all in model_config for next iteration
        model_config["scoring"] = updated_state_weights
        model_config["biases"] = updated_state_biases


        model_config["scoring_trans_row 0"] = updated_row_0_weights
        model_config["biases_trans_row 0"] = updated_row_0_biases

        model_config["scoring_trans_row 1"] = updated_row_1_weights
        model_config["biases_trans_row 1"] = updated_row_1_biases

        model_config["scoring_trans_row 2"] = updated_row_2_weights
        model_config["biases_trans_row 2"] = updated_row_2_biases



    standard_params = unpack_reparameterized_params(last_params)
    converged_hyperparams = {"scoring": model_config.get("scoring"), "biases": model_config.get("biases"),
                             "scoring_trans_row 0": model_config.get("scoring_trans_row 0"), "biases_trans_row 0": model_config.get("biases_trans_row 0"),
    "scoring_trans_row 1": model_config.get("scoring_trans_row 1"), "biases_trans_row 1": model_config.get("biases_trans_row 1"),
                             "scoring_trans_row 2": model_config.get("scoring_trans_row 2"), "biases_trans_row 2": model_config.get("biases_trans_row 2")}

    return v_predictions, standard_params, converged_hyperparams
def model_building(these_empirical_counts, these_empirical_kernels, these_initial_params, raw_distributions, severity_deltas, model_config):
    """Function that contains all the models fit to the observed data."""
    reparameterized = reparameterize(these_initial_params)
    this_fit_SSM = fit_latent_state_space_model(these_empirical_counts, these_empirical_kernels, reparameterized, raw_distributions,
                                                severity_deltas, model_config)
    return this_fit_SSM

def latent_state_clustering(predicted_latent_trajectory):
    """Function that applies clustering methods to the predicted latent state trajectory for
    later trajectory inference. Prunes and merges clusters that are too small."""
    clusters = KMeans(random_state=0)
    labels = clusters.fit_predict(predicted_latent_trajectory)
    regions = clusters.cluster_centers_
    regions_distances = distance_matrix(regions, regions) #distances between clusters
    counts = np.bincount(labels)

    #merging small clusters
    new_labels, active = merge_minimal_clusters(
        labels, counts, regions_distances, threshold=10
    )

    #re-indexing labels
    unique = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique)}
    labels = np.array([label_map[l] for l in new_labels])

    # Recompute centers and counts
    new_centers = recalculate_merged_cluster_centers(predicted_latent_trajectory, labels)
    counts = np.bincount(labels)

    return new_centers, labels, counts, regions_distances

def merge_minimal_clusters(labels, counts, Distances, threshold = 10):
    """Merges clusters with too few points by assigning them to the closest clusters."""
    labels = labels.copy()
    counts = counts.copy()
    cluster_numbers = len(counts)

    #active clusters
    active_clusters = np.ones(cluster_numbers, dtype = bool)
    for cluster_index in np.argsort(counts):
        if not active_clusters[cluster_index]: #skip inactive clusters until completing
            continue
        if counts[cluster_index] >= threshold:
            continue
        #excluding self and already inactive clusters for efficiency
        valid = np.where(active_clusters & (np.arange(cluster_numbers) != cluster_index))[0]
        if len(valid) == 0:
            continue
        nearest_cluster = valid[np.argmin(Distances[cluster_index, valid])]

        #reassigning labels
        labels[labels == cluster_index] = nearest_cluster

        # Update counts
        counts[nearest_cluster] += counts[cluster_index]
        counts[cluster_index] = 0
        active_clusters[cluster_index] = False

    return labels, active_clusters

def recalculate_merged_cluster_centers(latent_states, labels):
    """Reassigns points to corrected clusters, having recalculated the center."""
    centers = []
    latent_states = np.asarray(latent_states)
    for c in np.unique(labels):
        centers.append(latent_states[labels == c].mean(axis=0))
    return np.vstack(centers)

def compute_cluster_dirichlets(empirical_latent_state_trajectory, dirichlets_and_histories):
    """Function to compute Dirichlets for different clusters of latent states. Wraps latent_state_clustering above."""
    pruned_clusters = latent_state_clustering(empirical_latent_state_trajectory)
    cluster_and_dirichets = defaultdict(list)
    cluster_and_fixed = {}

    for index, latent_state in enumerate(empirical_latent_state_trajectory):
        corresponding_dirichlet = list(dirichlets_and_histories.values())[index]
        removed_prior = np.array([item-1 for item in corresponding_dirichlet])
        which_cluster = str(pruned_clusters[1][index])
        cluster_and_dirichets[which_cluster].append(removed_prior)

    for cluster, dirichlets in cluster_and_dirichets.items():
        added_dirichlets = np.sum(dirichlets, axis=0).tolist()
        cluster_and_fixed[cluster] = added_dirichlets + np.array([1, 1, 1]) #uninformative prior

    return cluster_and_fixed, pruned_clusters

def assign_label(x, centers):
    """Function to assign labels to latent states to a label in clustering."""
    x = np.asarray(x)
    centers = np.asarray(centers)
    return np.argmin(np.linalg.norm(centers - x, axis=1))
