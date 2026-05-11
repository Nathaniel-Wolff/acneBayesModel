from acne_model import data_parsing
from acne_model import process_data_and_build_HBM
import sys


all_names = sys.stdin.read()
all_names_split = all_names.split(",")
all_names_split_real = [name.strip() for name in all_names_split]

this_raw_data_name = all_names_split_real[0]
this_diet_raw_data_name = all_names_split_real[1]
json_name = all_names_split_real[2]

data_returns = data_parsing(this_raw_data_name, this_diet_raw_data_name)
these_frames = data_returns[0]
this_patients_diets = data_returns[1]
this_patients_treatments = data_returns[2]
this_patients_iAUCs = data_returns[3]
these_raw_frames = data_returns[4]

this_HBM = process_data_and_build_HBM(these_frames, model_priors_and_config_handle=json_name,
                                          these_raw_frames=these_raw_frames,
                                          patients_diet_data=this_patients_diets,
                                          patients_treatment_data=this_patients_treatments,
                                          patients_iAUCs=this_patients_iAUCs)
print(this_HBM)


