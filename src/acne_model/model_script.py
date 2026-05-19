from acne_model import data_parsing
from acne_model import process_data_and_build_HBM
from scipy.stats import multivariate_normal, gamma
import json
import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List, Dict, Tuple

@asynccontextmanager
async def train_model(app: FastAPI):
    print("🚀 [LIFESPAN] Container booted. Starting data parsing step...", flush=True)
    base_path = "/DockerAcneBayesModel/data"

    this_raw_data_name = os.path.join(base_path, "sim_acne_amended_v2.csv")
    this_diet_raw_data_name = os.path.join(base_path, "sim_acne_diet.csv")
    json_name = os.path.join(base_path, "bhm_model_config.json")

    data_returns = data_parsing(this_raw_data_name, this_diet_raw_data_name)
    print("📈 [LIFESPAN] Data parsed successfully. Initializing HBM optimization loop...", flush=True)

    model_params, treatment_params = process_data_and_build_HBM(
        data_returns[0],
        model_priors_and_config_handle=json_name,
        these_raw_frames=data_returns[4],
        patients_diet_data=data_returns[1],
        patients_treatment_data=data_returns[2],
        patients_iAUCs=data_returns[3])

    print("✅ [LIFESPAN] HBM training complete! Passing control back to Uvicorn.", flush=True)
    #caching params for later usage in run_model
    this_app.state.model_params = model_params
    this_app.state.treatment_params = treatment_params

    yield


this_app = FastAPI(lifespan=train_model)
class PatientDataRequest(BaseModel):
    """Request body capturing treatment series and initial latent state from the Shiny client. """
    initial_latent_state: List[float] #non-nested (3x1)
    treatment_series: List[List[float]] #nested (2xn) with n days of treatment

    class example():
        json_setup_example = {"example":
                                  {"initial_latent_state": [0.1, -0.2, 0.1],
                                   "treatment_series_3_days": [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
                                   }}

def parse_T_into_Tj(T_series_i, day_index, treatment_params, last_steady_state_concs):
    """Assumes first component is isotretinoin dosage and second is clindamycin-BPO."""
    clin = float(treatment_params["k_clin_bioa"]) * float(treatment_params["k_hydrolysis"]) * T_series_i[1] - \
           float(treatment_params["k_elim_clin"]) * last_steady_state_concs[0]
    bpo_rad_MM = (float(treatment_params["vmax_hlyt"]) * T_series_i[1]) / (T_series_i[1] + float(treatment_params["K_max"]))
    bpo_rad = float(treatment_params["k_bpo_bioa"]) * bpo_rad_MM - float(treatment_params["k_elim_BPO"]) * last_steady_state_concs[1]

    isotret = 0
    compartment_dif = float(treatment_params["k_12"]) / (float(treatment_params["alpha"]) - float(treatment_params["beta"]))
    for delta_t in range(day_index):
        increase = (compartment_dif * np.exp(float(treatment_params["beta"]) * delta_t) - np.exp(float(treatment_params["alpha"]) * delta_t)) * T_series_i[0]
        isotret += increase

    return np.array([clin, bpo_rad, isotret]).reshape(-1,1)


def run_model(F, T_series, G_j, M, b, Q, alpha, initial_z, treatment_params, initial_steady_state_concs =
              np.array([0.0, 0.0, 0.0])):
    """Uses the imputed biomarker marginalized state transition matrix, F, the patient specific treatment effect matrix G_j,
    and the process covariance matrix Q to calculate the latent state series via z_{t} = F @ z_{t-1} + G_j @ T_j + e_t{t},
    where e ~ N(0, Q). T_j refers to the patient's treatment matrix."""
    last_z = np.array(initial_z).reshape(-1, 1)
    T_series = np.array(T_series)

    gammas_params = []
    CIs = {}

    ss_concs = np.array(initial_steady_state_concs, dtype=float)

    for latent_state_index in range(len(T_series)):
        process_noise = multivariate_normal.rvs(mean=np.zeros(F.shape[0]), cov=Q).reshape(-1, 1)
        T_i = T_series[latent_state_index]
        parsed_T_i = parse_T_into_Tj(T_series_i=T_i, day_index=latent_state_index, treatment_params=treatment_params,
                                     last_steady_state_concs=ss_concs)
        ss_concs = parsed_T_i.flatten()

        next_latent = F @ last_z + G_j @ parsed_T_i + process_noise
        last_z = next_latent

        mean = np.exp(M @ next_latent + b)[0, 0] #ensuring scalar
        beta = alpha / mean
        gammas_params.append([float(alpha), float(beta)])

        #outputting the 95% credible intervals of each distribution
        lower, upper = gamma.ppf([0.025, 0.975], a=alpha, loc=0, scale=1/beta)
        CIs[str(latent_state_index)] = [float(mean), float(lower), float(upper)]

    return CIs, gammas_params

@this_app.post("/send_predictions")
async def model_predict(Data: PatientDataRequest):
    """Runs the relevant calculations to predict treatment series."""
    params = this_app.state.model_params
    treat_params = this_app.state.treatment_params

    t_arr = np.array(Data.treatment_series)
    g_arr = np.array(params["Treatment_Mapping_G"])

    # This will print directly to your running terminal window!
    print(f"📐 [DEBUG MATH] G matrix shape: {g_arr.shape}", flush=True)
    print(f"📐 [DEBUG MATH] Incoming T_series shape: {t_arr.shape}", flush=True)
    print(f"📐 [DEBUG MATH] Single slice T[0] shape: {t_arr[0].shape if len(t_arr.shape) > 1 else t_arr.shape}",
          flush=True)

    CIs, gammas_params = run_model(F = np.array(params["LS_Evolution_F"]), T_series = Data.treatment_series,
                                   treatment_params=treat_params,
                                   G_j = np.array(params["Treatment_Mapping_G"]),
                                   M = np.array(params["M_linking_func"]), b = np.array(params["b_linking_func"]),
                                   Q = np.array(params["Process_Cov"]),
                                   alpha = float(params["alpha_linking_func"]), initial_z=Data.initial_latent_state)
    return {"credible_intervals": CIs}






