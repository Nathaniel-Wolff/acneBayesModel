#for training of actual model (continous Latent State Space/BHM)

import pandas as pd
import numpy as np
from numpy.linalg import pinv
from collections import defaultdict
from functools import reduce
import json
from scipy.stats import invwishart, multivariate_normal
from scipy.linalg import eigvals, solve_continuous_lyapunov
from scipy.stats import entropy
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import jax.numpy as jnp
from jax import jacfwd

def layer_2_bml_observable_biomarkers(design_matrix, imputed_biomarkers_matrix, current_mean_matrix, prior_precision_matrix, prior_scale_matrix, dof):
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
    post_mean_matrix = np.linalg.inv(post_precision_matrix) @ \
                       (design_matrix.T @ imputed_biomarkers_matrix + prior_precision_matrix @ current_mean_matrix)
    converted_design_matrix = design_matrix

    residuals = imputed_biomarkers_matrix - (converted_design_matrix @ post_mean_matrix)
    mean_diff = post_mean_matrix - current_mean_matrix
    #ensuring post scale matrix is PSD
    post_scale_matrix = prior_scale_matrix + \
                        (residuals.T @ residuals) + \
                        (mean_diff.T @ prior_precision_matrix @ mean_diff)

    post_dof = dof + len(post_mean_matrix.T)


    first_layer_params = {"precision_matrix": post_precision_matrix, "post_mean_matrix": post_mean_matrix,
              "post_scale_matrix": post_scale_matrix, "post_dof": post_dof, "design_matrix": design_matrix, "imputed_biomarkers_matrix": imputed_biomarkers_matrix}

    return first_layer_params
def mask_A_mechanistically(A_in):
    """Function used to mask entries of A, the mapping from imputed biomarkers to latent states.
     Uses mechanistic hypotheses to do so."""
    #Row 1 -Which imputed biomarkers are the terms in the bacterial dysbiosis part of the latent state. Should be the last 2.
    #Row 2 - Which imputed biomarkers are the terms in the inflammation part of the latent state. Should be the last 2.
    #---To distinguish Rows 1/2, each will have slightly different couplings based on literature
    #Row 3 - For sebum effect (based on expression in the patient's keratinocytes)

    A = A_in.copy()

    A[0, 0] = 0.0  # mTORC1 ! in Bacterial
    A[0, 1] = 0.0  # mTORC1 ! in Inflammatory
    A[2, 2] = 0.0  # Lipase ! in Sebum

    #Mechanistic boosts
    A[2, 0] = A[2, 0] * 1.25  # Bacterial driver
    A[1, 1] = A[1, 1] * 1.25  # Inflammatory driver
    A[0, 2] = A[0, 2] * 1.25  # Sebum driver

    return A
def maximization_compute_cross_covs(latent_states, latent_state_covs, latent_states_lagged_covs):
    #computing state correlations and serial lag correlations at the same time

    dim = latent_states[0].shape[0]
    state_correlations = np.zeros((dim, dim))
    one_lag_correlations = np.zeros((dim, dim))
    prev_state_correlations = np.zeros((dim, dim))

    #print(len(latent_states), len(latent_state_covs), len(latent_states_lagged_covs))

    for t in range(1, len(latent_states)):
        z_t = latent_states[t]
        z_prev = latent_states[t - 1]

        P_t = latent_state_covs[t]
        P_prev = latent_state_covs[t - 1]

        # Use t-1 to access the lag-covariance list
        P_lag = latent_states_lagged_covs[t - 1]

        state_correlations += (z_t @ z_t.T + P_t)
        prev_state_correlations += (z_prev @ z_prev.T + P_prev)
        one_lag_correlations += (z_t @ z_prev.T + P_lag)


    clamped_state_correlation = eigenclamp_sym_enforce(state_correlations)
    clamped_one_lag_correlations = eigenclamp_sym_enforce(one_lag_correlations)
    clamped_prev_state_correlations = eigenclamp_sym_enforce(prev_state_correlations)


    return {"state_correlations": clamped_state_correlation,
        "one_lag_correlations": clamped_one_lag_correlations,
        "prev_state_correlations": clamped_prev_state_correlations}
def eigenclamp_sym_enforce(M, floor = 1e-10):
    #----forcing Symmetry
    smoothed_M = 0.5 * (M + M.T)

    #----eigen-Clamping
    vals, vecs = np.linalg.eigh(smoothed_M)
    vals = np.maximum(vals, floor)  # Clamp to a tiny positive floor
    fully_smoothed_M = vecs @ np.diag(vals) @ vecs.T

    return fully_smoothed_M
def compute_cross_layer_moments(layer_2_mapping, all_within_layers_moments, first_layer_params):
    """Also solves for the latent state serial covariance, Cov(v_t, v_t-1), using OU process params."""
    here_1st_precision_matrix = first_layer_params["precision_matrix"]
    here_2nd_precision_matrix = np.linalg.inv(all_within_layers_moments["2nd_layer_posterior_covariance"])

    joint_covariance_z_y = np.array([here_1st_precision_matrix + layer_2_mapping.T @ here_2nd_precision_matrix @ layer_2_mapping, -layer_2_mapping.T @ here_2nd_precision_matrix],
                                [-here_2nd_precision_matrix @ layer_2_mapping, here_2nd_precision_matrix])
    cross_layer_moment = joint_covariance_z_y + all_within_layers_moments["latent_state_forward_estimate"] @ all_within_layers_moments["posterior_imputed_mean"].T

    cross_layer_lagged_moments = {"joint_covariance_z_y": joint_covariance_z_y, "cross_layer_moment": cross_layer_moment }

    return cross_layer_lagged_moments
def layer_3_imputed_biomarkers_multivar_model(first_layer_params, this_last_observables_cov_fixed, this_last_mean, this_last_coeff_matrix, ou_matrices, current_ls_ev_F, current_imp_map_A, last_initial_state, current_process_noise_cov, current_measurement_noise_cov, Q_prior,
                                              tikh_reg_term):
    """Fits layer 2 of the imputed biomarker multivariate model. Implements one iteration of EM."""
    # ends with learning the relative contributions of the imputed biomarkers to total bacterial dysbiosis, total sebum production, total inflammation
    usable_imputeds_matrix = first_layer_params["imputed_biomarkers_matrix"]
    #---Expectation Step
    expectation_results = all_layers_expectation_step(last_ls_ev_F=current_ls_ev_F, last_imp_map_A= current_imp_map_A, last_process_noise_cov=current_process_noise_cov, last_measurement_noise_cov=current_measurement_noise_cov,
                                                      imputed_biomarkers_matrix=usable_imputeds_matrix, last_initial_latent_state=last_initial_state, last_process_cov= this_last_observables_cov_fixed)

    unpacked_latent_states  = expectation_results["smoothed_latent_states"]
    latent_state_covs = expectation_results["smoothed_latent_state_covs"]
    latent_state_lagged_covs = expectation_results["lagged_latent_state_covs"]
    latent_state_resids = expectation_results["latent_state_resids"]
    these_innovations = expectation_results["innovations"]
    these_innovation_covs = expectation_results["innovation_covs"]


    #---Maximization Step
    this_maximization = full_maximization_step_HBM(latent_states = unpacked_latent_states, first_layer_params=first_layer_params,
                                                   smoothed_covariances = latent_state_covs, lagged_covariances = latent_state_lagged_covs, imputed_biomarkers=usable_imputeds_matrix, innovations=these_innovations, innovation_covs=these_innovation_covs, initial_Q_prior=Q_prior,
                                                   given_tikh_reg=tikh_reg_term)
    #maximized_params = {"current_ls_ev_F": new_ls_ev_F, "current_impt_map_H": new_imp_map_H,
                        #"current_Q": new_process_cov_Q, "current_R": new_measurement_cov_R}

    #---Computation of Ornstein Uhlenbeck RV, indexed by its continous Lyanpunov equation solution matrix
    #this_OU_State_Matrix_const = ou_matrices["State_Matrix_OU"]
    #this_OU_State_Matrix = this_OU_State_Matrix_const * np.identity(cov_matrix_maximized.shape[0]) #check dimensions for Wiener process and State Matrix

    #this_OU_Weiner_Matrix_const = ou_matrices["Wiener_Matrix_OU"]
    #this_OU_Weiner_Matrix = this_OU_Weiner_Matrix_const * np.identity(cov_matrix_maximized.shape[0])
    #mental note: make sure covariance matrix for second normal is initialized and also the ones for the OU processes above
    #this_OU_index_matrix, test_result = multivar_Ornstein_Uhlenbeck(state_matrix=this_OU_State_Matrix, diffusion_matrix=this_OU_Weiner_Matrix)


    #final_cov_matrix_maximized = cov_matrix_maximized + this_OU_index_matrix

    return this_maximization
def backpropagate_KG_to_layer_2(design_matrix, innovations, innovation_covs, last_W, learning_rate = 1e-5):
    first_grad_term = np.linalg.pinv(innovation_covs[0]) @ innovations[0].reshape(-1, 1) @ design_matrix[0].reshape(1, -1)

    ll_grad_wrt_layer_1 = np.zeros((first_grad_term.shape[0], first_grad_term.shape[1]))
    for t, observables_row in enumerate(design_matrix):
        single_grad_term = np.linalg.pinv(innovation_covs[t]) @ innovations[t].reshape(-1, 1) @ observables_row.reshape(1, -1)
        ll_grad_wrt_layer_1 += single_grad_term

    new_W = last_W - learning_rate * ll_grad_wrt_layer_1.T
    return new_W
def all_layers_expectation_step(imputed_biomarkers_matrix, last_ls_ev_F, last_imp_map_A, last_initial_latent_state, last_process_cov,
                                last_process_noise_cov, last_measurement_noise_cov, ode_jacobian, epsilon = 1e-9):
    #computing first and second moments for layer 2
    #conducts all k Kalman filter/smoothings for the latent biological state z_{t} = [B_t, I_t, S_t]
    """Follows this latent state evolution model: z_{t+1) = F(z_{t}) + B(u_{t}) + w_{k}; w_{k} ~ N(0, Q)
    and this imputed biomarker to latent state mapping: z_{t} = A(y_{t}) + v_{k};  v_{k} ~ N(0, R).
    """
    a_priori_latent_states = []
    a_priori_covs = []
    a_post_covs = []
    a_post_latent_states = []
    innovations = []

    latent_state_covs = []

    innovation_covs = []
    latent_state_resids = []
    last_a_post_estimate = last_initial_latent_state
    last_a_post_cov_est = last_process_cov

    kalman_gains = []
    smoothing_gains = []

    #calculating intercept term for predicted imputed biomarkers, ensuring scales of latent states and imputed biomarkers don't blow up covariance
    intercept = np.mean(imputed_biomarkers_matrix, axis=0)
    #print("Matrix Shape:", imputed_biomarkers_matrix.shape)
    #print("Matrix Max/Min:", np.max(imputed_biomarkers_matrix), np.min(imputed_biomarkers_matrix))

    for iteration in range(0, len(imputed_biomarkers_matrix)):
        #---Prediction Step
        a_priori_state = last_ls_ev_F @ last_a_post_estimate #+B_k @ U_k
        a_priori_latent_states.append(a_priori_state)
        a_priori_cov = last_ls_ev_F @ last_a_post_cov_est @ last_ls_ev_F.T + last_process_noise_cov
        a_priori_covs.append(a_priori_cov)
        #---Update Step
        pred_imputed = last_imp_map_A @ a_priori_state + intercept #removed random noise realization, adding intercept term

        innovation = imputed_biomarkers_matrix[iteration] - pred_imputed
        innovation_cov = last_imp_map_A @ a_priori_cov @ last_imp_map_A.T + last_measurement_noise_cov
        innovation_cov += np.eye(innovation_cov.shape[0]) * epsilon #Tikhonov regularization to improve convergence

        opt_Kal_gain = a_priori_cov @ last_imp_map_A.T @ np.linalg.pinv(innovation_cov) #corrected
        kg_size = opt_Kal_gain.shape

        a_post_state  = a_priori_state + opt_Kal_gain @ innovation
        #changed to using Symmetrized Joseph Form to improve stability
        IKH = np.identity(kg_size[0]) - opt_Kal_gain @ last_imp_map_A
        a_post_cov = IKH @ a_priori_cov @ IKH.T + opt_Kal_gain @ last_measurement_noise_cov @ opt_Kal_gain.T
        a_post_covs.append(a_post_cov)
        a_post_residual = pred_imputed - last_imp_map_A @ a_post_state

        #computing smoothing gain explicitly
        if iteration < len(imputed_biomarkers_matrix) - 1:
            smoothing_gain =  last_a_post_cov_est @ last_ls_ev_F.T @ np.linalg.pinv(a_priori_cov) #pinverse for now, need to fix singularity
            smoothing_gains.append(smoothing_gain)

        a_post_latent_states.append(a_post_state)
        latent_state_covs.append(a_post_cov)
        latent_state_resids.append(a_post_residual)
        innovations.append(innovation)
        innovation_covs.append(innovation_cov)
        kalman_gains.append(opt_Kal_gain)

        #updating
        last_a_post_estimate = a_post_state
        last_a_post_cov_est = a_post_cov

    #Recursive RTS smoothing initialization
    T = len(a_post_latent_states)
    last_lag = smoothing_gains[T - 2] @ latent_state_covs[T-1]
    lagged_latent_state_covs = [last_lag]
    smoothed_latents = [a_post_latent_states[-1]]
    smoothed_covs = [a_post_covs[-1]]

    #Completing RTS smoothing of latent states and covariances (corrected)
    for t in range(T-2, -1, -1):
        #---lagged-1 latent state covariance smoothing
        if t > 0:
            current_Gain = smoothing_gains[t]
            last_Gain = smoothing_gains[t-1]
            res_term = lagged_latent_state_covs[0] - last_ls_ev_F @ latent_state_covs[t]
            new_lagged_cov = latent_state_covs[t] @ last_Gain.T + current_Gain @ res_term @ last_Gain.T
            lagged_latent_state_covs.insert(0, new_lagged_cov)
        #----latent state smoothing plus enforcing symmetry
        innovation_latent_state = smoothed_latents[0] - last_ls_ev_F @ a_post_latent_states[t]
        smoothed_latent = a_post_latent_states[t] + smoothing_gains[t] @ innovation_latent_state
        smoothed_latents.insert(0, smoothed_latent)
        #----covariance smoothing
        smoothed_cov = a_post_covs[t] + smoothing_gains[t] @ (smoothed_covs[0] - a_priori_covs[t+1]) @ smoothing_gains[t].T
        smoothed_cov_sym_enforced = 0.5 * (smoothed_cov + smoothed_cov.T)
        vals, vecs = np.linalg.eigh(smoothed_cov_sym_enforced)
        vals = np.maximum(vals, 1e-10)  #clamping to a small positive floor to ensure positive eigenvalues
        smoothed_cov_clipped = vecs @ np.diag(vals) @ vecs.T
        smoothed_covs.insert(0, smoothed_cov_clipped)


    results = {"smoothed_latent_states": smoothed_latents, "smoothed_latent_state_covs": smoothed_covs, "latent_state_resids": latent_state_resids, "kalman_gains": kalman_gains, "smoothing_gains": smoothing_gains,
               "lagged_latent_state_covs": lagged_latent_state_covs, "innovations": innovations , "innovation_covs": innovation_covs}

    return results
def full_maximization_step_HBM(latent_states, smoothed_covariances, lagged_covariances, imputed_biomarkers, first_layer_params, innovations, innovation_covs, initial_Q_prior, given_tikh_reg, EM_learning_rate = 0.1, prior_weight = .9):
    """Maximizes in closed form and with Gradient Descent."""
    sufficient_stats = maximization_compute_cross_covs(latent_states = latent_states,
                                                       latent_state_covs = smoothed_covariances, latent_states_lagged_covs = lagged_covariances)
    #added Tikhonov regularization term and learning rate
    #making sure it is scaled relative the trace of the prev state correlations
    trace_val = np.trace(sufficient_stats["prev_state_correlations"])
    tikh_reg_term = max(1e-2, 0.01 * (trace_val / 3)) * np.eye(3)

    new_ls_ev_F = sufficient_stats["one_lag_correlations"] @ np.linalg.pinv(
        sufficient_stats["prev_state_correlations"] + tikh_reg_term)

    outer_products = [np.outer(imp, lat) for lat, imp in zip(latent_states, imputed_biomarkers)]

    sum_outer = np.sum(outer_products, axis=0)
    new_imp_map_A = sum_outer @ np.linalg.pinv(sufficient_stats["state_correlations"] + tikh_reg_term)
    new_process_cov_Q_MLE = 1/len(latent_states) * (sufficient_stats["state_correlations"] - new_ls_ev_F @ sufficient_stats["one_lag_correlations"].T)
    #changing from pure MLE to MAP Estimate for numerical stability
    new_process_cov_Q = (1 - prior_weight) * new_process_cov_Q_MLE + prior_weight * initial_Q_prior
    new_measurement_cov_R = np.zeros(new_imp_map_A.shape)
    T = len(latent_states)
    for latent_state, imp_vector, smoothed_covariance in zip(latent_states, imputed_biomarkers, smoothed_covariances):
        one_term = np.outer(imp_vector - new_imp_map_A @ latent_state, (imp_vector - new_imp_map_A @ latent_state).T) + new_imp_map_A @ smoothed_covariance @ new_imp_map_A.T
        new_measurement_cov_R += one_term

    new_measurement_cov_R /= T #fixed with normalization
    new_W = backpropagate_KG_to_layer_2(design_matrix=first_layer_params["design_matrix"], innovations=innovations, innovation_covs=innovation_covs, last_W=first_layer_params["post_mean_matrix"])

    # forcing symmetry and positivity to improve convergence
    new_ls_ev_F_enforced = 0.5 * (new_ls_ev_F + new_ls_ev_F.T)
    new_ls_ev_F_enforced += 1e-6 * np.eye(new_ls_ev_F.shape[0])
    damped_ls_ev_F_enforced = (1 - EM_learning_rate) * new_ls_ev_F_enforced + EM_learning_rate * new_ls_ev_F_enforced

    #checking eigenvalues of F and dividing by the largest one if needed
    is_unstable, radius = check_spectral_radius(damped_ls_ev_F_enforced)
    if is_unstable:
        damped_ls_ev_F_enforced = damped_ls_ev_F_enforced / (radius + 1e-6)

    damped_imp_map_A = (1 - EM_learning_rate) * new_imp_map_A + EM_learning_rate * new_imp_map_A

    #enforcing symmetry
    new_process_cov_Q_enforced = 0.5 * (new_process_cov_Q + new_process_cov_Q.T)
    new_process_cov_Q_enforced += 1e-2 * np.eye(new_process_cov_Q.shape[0])
    final_enforced_Q = force_positive_definite(new_process_cov_Q_enforced)
    #adding diagonal constraints to Q and R to keep them from converging
    new_Q = np.diag(np.diag(final_enforced_Q))

    new_measurement_cov_R_enforced = 0.5 * (new_measurement_cov_R + new_measurement_cov_R.T)
    new_measurement_cov_R_enforced += 1e-2 * np.eye(new_measurement_cov_R.shape[0])
    final_enforced_R = force_positive_definite(new_measurement_cov_R_enforced)
    new_R = np.diag(np.diag(final_enforced_R))

    maximized_params = {"current_ls_ev_F": damped_ls_ev_F_enforced, "current_imp_map_A": damped_imp_map_A,
                        "current_Q": new_Q, "current_R": new_R,
                        "current_W": new_W, "smoothed_latent_states": latent_states, "smoothed_latent_state_covs": smoothed_covariances}

    return maximized_params
def layer_4_latent_state_severity_mapping(maximized_mean, treatment_series, diet_data, weights = np.array, biases = np.array):
    latent_state = treatment_diet_mapping()
    mean = np.exp(weights @ latent_state + biases)


    #need to use the normal posterior mean from layer 2 as the likelihood for the gamma
def check_spectral_radius(A):
    """To improve numerical instability, this function calculates the spectral radius of one of the mappings used.
    Flags said mapping if eigenvalues >1 are found in it."""
    #calculates absolute values in case eigenvalues are negative/complex
    eigenvals = np.linalg.eigvals(A)
    abs_eigenvals = np.abs(eigenvals)

    max_radius = np.max(abs_eigenvals)
    is_unstable = max_radius > 1.0

    return is_unstable, max_radius
def force_positive_definite(M, floor=1e-4):
    """Used to improve numerical instability with clipping of eigenvals to a positive floors."""
    M = 0.5 * (M + M.T)
    vals, vecs = np.linalg.eigh(M)
    vals = np.maximum(vals, floor)
    return vecs @ np.diag(vals) @ vecs.T
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
    #attempting to drop Bulk Androgen Concentration
    #observables_names = ['Insulin Conc.', 'IGF1 Conc.', 'Skin pH', 'NLR']
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

    #added here, the feedback term is residualized against IGF1 concentrations itself (finding feedback not due to Androgen alone, in case of high correlation)
    androgen_feedback_linear_model = LinearRegression().fit(raw_design_matrix[["IGF1 Conc."]], raw_design_matrix["IGF1-Androgen Feedback"])
    second_coord = raw_design_matrix["IGF1-Androgen Feedback"] - androgen_feedback_linear_model.predict(raw_design_matrix[["IGF1 Conc."]])
    raw_design_matrix["IGF1-Androgen Feedback"] = second_coord
    #renormalizing the result
    raw_design_matrix["IGF1-Androgen Feedback"] = (raw_design_matrix["IGF1-Androgen Feedback"] - raw_design_matrix[
        "IGF1-Androgen Feedback"].mean()) / raw_design_matrix["IGF1-Androgen Feedback"].std()


    #outputting result of GMM on pH here. Uses hyperparams that can be configured by the user.
    gaussian_evaluated = [gaussian_weights[i]*univariate_gaussian(mean=this_mean, std_dev=this_std, dof=raw_design_matrix["Skin pH"]) for i, (this_std, this_mean) in enumerate(gaussian_params)]
    gaussian_output = np.sum(gaussian_evaluated, axis = 0)
    raw_design_matrix["pH GMM"] = gaussian_output

    drop_columns = ["Insulin Conc.", "IGF1 Conc.", "Bulk Androgen Conc.", "Skin pH"]
    final_raw_design_matrix = raw_design_matrix.drop(columns = drop_columns)

    #print(final_raw_design_matrix)

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
def normalize_matrix_columnwise(M):
    return (M - np.mean(M, axis=0)) / np.std(M, axis=0)



def ur_isotret(k_12, alpha, beta, t):
    """Function to calculate isotretinoin unit response in bioavaiable concentration relative to 1 gram dose."""
    slope = k_12/(alpha - beta)
    return slope * (np.exp(-beta * t) - np.exp(-alpha * t)) #t is in hours, discretized to 12 hour periods

def find_C2_isotret(k_12, alpha, beta, dose_amt, dt): #dt is in hours, discretized to 12 hour periods
    """Function to calculate current peripheral concentration of isotretinoin via the unit response above."""

    return dose_amt * ur_isotret(k_12, alpha, beta, dt)
def treatment_diet_mapping(params, T_raw, PPGR_trained, diet_data, c_leu, current_time = 0, ss_concs = np.zeros(3)):
    """Function mapping raw treatment dosages and diet covariate to latent state, V(T_j, Phi_{i, j})."""
    #T_raw is previously zipped with treatment names in order.
    #need to account for molar ratios here since using gram weights with percentages. Do this later.
    k_clin_bioa, k_hydrolysis, k_elim_clin  = params["k_clin_bioa"], params["k_hydrolysis"], params["k_elim_clin"]
    k_bpo_bioa, vmax_hlyt, K_max, k_elim_BPO =  params["k_bpo_bioa"], params["vmax_hlyt"], params["K_max"], params["k_elim_BPO"]
    k_12, alpha, beta, = params["k_12"], params["alpha"], params["beta"]


    bioa_clin = k_clin_bioa * k_hydrolysis * T_raw["ClPhos-BPO (g)"] - k_elim_clin * ss_concs["Clin"]
    bioa_BPO = k_bpo_bioa * ( (vmax_hlyt * T_raw["ClPhos-BPO (g)"]) / (K_max + T_raw["ClPhos-BPO (g)"])) - k_elim_BPO * ss_concs["BPO"]
    bioa_isotret = find_C2_isotret(k_12 = k_12, alpha = alpha, beta = beta, dose_amt=T_raw["Isotret (g)"], dt = current_time)

    now_bioa_clin = ss_concs["Clin"] + bioa_clin
    now_bioa_BPO = ss_concs["BPO"] + bioa_BPO
    now_bioa_isotret = ss_concs["Isotret"] + bioa_isotret

    T_j = np.array( [now_bioa_clin, now_bioa_BPO, now_bioa_isotret])
    PPGR_effective = PPGR_trained.predict(diet_data)
    phi = [PPGR_effective, c_leu]

    return T_j, phi
def treatment_diet_mapping_jacobian(c_clind_SS, c_clind_phos, T_raw, params, ss_concs):
   jacobian = jacfwd(treatment_diet_mapping, argnums=0)
   return jacobian



def fit_HBM_model(initial_severities_matrix, initial_design_matrix, initial_biomarkers_matrix,
                  prior_mean_matrix, prior_precision_matrix, prior_scale_matrix, initial_2nd_layer_cov_mat, initial_2nd_layer_mean_mat,
                  initial_2nd_layer_coeff_mat, last_OU_matrices, initial_F, initial_A, initial_Q, initial_R, all_diet_data, all_treatment_data, all_iAUCs,
                  dof = 0, max_epochs = 20, tikh_reg_term = 1e-2, learning_rate = 1e-5):

    #Initialization of all model constants already completed in model config

    ##----Initializing the EM Loop, with normalization
    last_design_matrix = normalize_matrix_columnwise(initial_design_matrix.to_numpy())
    last_biomarkers_matrix = normalize_matrix_columnwise(initial_biomarkers_matrix.to_numpy())
    last_mean_matrix = prior_mean_matrix
    last_severities_matrix = initial_severities_matrix #already normalized

    last_2nd_layer_covariance_matrix = initial_2nd_layer_cov_mat #initialized but not updated explicitly
    last_2nd_layer_mean_matrix = initial_2nd_layer_mean_mat #initialized but not updated explicitly
    last_2nd_layer_coeff_matrix = initial_2nd_layer_coeff_mat #initialized but not updated explicitly

    #updated explicitly
    last_F = initial_F

    #breaking symmetry in A at initialization
    initial_A[2, 0] = 0.8  # Lipase -> Bacterial
    initial_A[1, 1] = 0.8  # Inflamm -> Inflamm
    initial_A[0, 2] = 0.8  # mTORC1 -> Sebum
    #slight random coupling noise  is added to the other drivers of A
    initial_A += np.random.normal(0, 0.05, (3, 3))
    #immediate masking
    last_A = mask_A_mechanistically(initial_A)

    last_Q = initial_Q
    last_R = initial_R

    processed_last_OU_matrices = {"State_Matrix_OU": last_OU_matrices["State_Matrix_OU"] * np.identity(last_biomarkers_matrix.shape[1]),
                                  "Wiener_Matrix_OU": last_OU_matrices["Wiener_Matrix_OU"] * np.identity(last_biomarkers_matrix.shape[1])}

    this_initial_latent_state = np.linalg.pinv(initial_A) @ last_biomarkers_matrix[0]
    this_initial_covariance = np.eye(3) * 0.1 #fixed

    PPGR_predictor_GBRTs = defaultdict(GradientBoostingRegressor)

    # ----Gradient Boosted Regression Tree for Inference of PPGR from diet data
    # Currently frozen for now, later will be optimized with rest

    for i, one_patient_diet_data in enumerate(all_diet_data.values()):
        diet_data_concat = np.concatenate(one_patient_diet_data, axis=0)

        this_GBRT = GradientBoostingRegressor()
        patient_iAUCs = np.array(all_iAUCs[i]).ravel()

        this_GBRT.fit(X=diet_data_concat, y=patient_iAUCs)
        PPGR_predictor_GBRTs[list(all_diet_data.keys())[i]] = this_GBRT.fit(X=diet_data_concat,
                                                                            y=patient_iAUCs)  # maintaining patient ID

    #----EM Loop
    for em_epoch in range(max_epochs):
        #----Kalman Filter
        one_iter_layer1_params = layer_2_bml_observable_biomarkers(design_matrix=last_design_matrix, imputed_biomarkers_matrix=last_biomarkers_matrix,
                                                                   current_mean_matrix=last_mean_matrix, prior_precision_matrix=prior_precision_matrix, prior_scale_matrix=prior_scale_matrix, dof = dof)
        one_iter_layer2_params = layer_3_imputed_biomarkers_multivar_model(first_layer_params=one_iter_layer1_params, this_last_observables_cov_fixed=this_initial_covariance, ou_matrices=processed_last_OU_matrices,
                                                                           this_last_mean=last_2nd_layer_mean_matrix, this_last_coeff_matrix=last_2nd_layer_coeff_matrix, current_ls_ev_F=last_F, current_imp_map_A=last_A,
                                                                           current_measurement_noise_cov=last_Q, current_process_noise_cov=last_R, last_initial_state=this_initial_latent_state, Q_prior = initial_Q, tikh_reg_term = tikh_reg_term)

        last_mean_matrix = one_iter_layer2_params["current_W"]
        last_F = one_iter_layer2_params["current_ls_ev_F"]

        last_A = mask_A_mechanistically(one_iter_layer2_params["current_imp_map_A"]) #included mechanistic masking/boosting

        last_Q = one_iter_layer2_params["current_Q"]
        last_R = one_iter_layer2_params["current_R"]

        this_initial_latent_state = one_iter_layer2_params["smoothed_latent_states"][0]
        this_initial_covariance = one_iter_layer2_params["smoothed_latent_state_covs"][0]


    converged_params = {"1st_Layer_Coeffs": last_mean_matrix, "LS_Evolution_F": last_F, "Imputed_Mapping_A": last_A,
                        "Process_Cov": last_Q, "Measurement_Cov": last_R}


    return converged_params
def process_data_and_build_HBM(these_dataframes, model_priors_and_config_handle, these_raw_frames, patients_diet_data, patients_treatment_data, patients_iAUCs):
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

    initial_imp_map_A = np.diag(model_priors_and_config["Initial_IMP_Map_A"] * np.ones(this_pmm.shape[1])) + np.random.normal(0, 0.1, (this_pmm.shape[1], this_pmm.shape[1]) )  #attempting to reduce linear independence

    initial_q_process_noise = model_priors_and_config["Initial_Q_Process_Noise"] * np.ones(this_pmm.shape[1]) #changed
    initial_r_measurement_noise = model_priors_and_config["Initial_R_Measurement Noise"] * np.ones(this_2nd_mean.shape[0])

    last_OU_matrices = {"State_Matrix_OU": model_priors_and_config["State_Matrix_OU"], "Wiener_Matrix_OU": model_priors_and_config["Wiener_Matrix_OU"]}  # placeholder

    this_fit_model = fit_HBM_model(initial_severities_matrix = this_ism, initial_design_matrix=this_ids, initial_biomarkers_matrix=this_ibm,
                                   prior_mean_matrix=this_pmm, prior_precision_matrix=this_ppm, prior_scale_matrix=this_pcm,
                                   dof = prior_dof, initial_2nd_layer_cov_mat=this_2nd_cov, initial_2nd_layer_mean_mat= this_2nd_mean,
                                   initial_2nd_layer_coeff_mat=this_2nd_coeff_matrix, last_OU_matrices=last_OU_matrices,
                                   initial_F=initial_ls_ev_f, initial_A=initial_imp_map_A, initial_Q=initial_q_process_noise, initial_R=initial_r_measurement_noise,
                                   all_diet_data=patients_diet_data, all_treatment_data=patients_treatment_data, all_iAUCs=patients_iAUCs)

    return this_fit_model
