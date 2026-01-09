#for raw data analysis functions

#imports
import matplotlib
import random
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle as rect
from itertools import permutations
from colorsys import rgb_to_hls, hls_to_rgb
import pandas as pd
import scipy as sp
from scipy import optimize
import numpy as np
import seaborn as sns
import copy
from collections import defaultdict, Counter
from matplotlib.cm import viridis
import statsmodels.api as sm
from scipy.stats import dirichlet
from scipy.stats import beta
from scipy.special import gammaln, psi
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KernelDensity
import json
import filterpy as fp
import numdifftools as nd
from functools import partial

def seperate_patients(raw_data):
    """Function that separates the raw dataframe via the following:
    1) Constructs an array tracking where the original date (2018-01-01) recurs.
    2) Uses that array to split raw_data into seperate dataframes.
    """
    # splitting data into different patients using leftmost column index (this is an assumption however)
    seperatePatientsIndices = list(raw_data.index[raw_data["date"] == "2018-01-01"])
    seperatePatientsIndices.append(len(raw_data))

    # checking to see if, after the same number of days in all dataframes, treatment of some sort was introduced
    # verifies that on day 29
    # then adds the days each additional treatment was added after the fact (turns out they are all the same too)
    allPatientsIntroDays = []

    # seperating single dataframe into list of dataframes for each patient
    startIndex = 0
    seperatePatientsDFs = []
    for endIndex in seperatePatientsIndices[1:]:
        seperatePatientsDFs.append(raw_data[startIndex:endIndex])
        startIndex = endIndex

    for seperatePatientDF in seperatePatientsDFs:
        treatmentIntroDays = []
        currentTreatment = seperatePatientDF["treatment"].iloc[0]

        for lineIndex in range(1, len(seperatePatientDF)):
            if seperatePatientDF["treatment"].iloc[lineIndex] != currentTreatment:
                treatmentIntroDays.append([currentTreatment, "end day is", lineIndex])
                currentTreatment = seperatePatientDF["treatment"].iloc[lineIndex]

        # remembering the end day of the last treatment and appending to the list
        lastTreatment = seperatePatientDF["treatment"].iloc[len(seperatePatientDF) - 1]
        treatmentIntroDays.append([lastTreatment, "end day is", len(seperatePatientDF) - 1])

        allPatientsIntroDays.append(treatmentIntroDays)

    return seperatePatientsDFs, allPatientsIntroDays

def add_history_metadata(seperatePatientsDFs, allPatientsIntroDays):
    """Function that adds a treatment history metadata column to each patient's dataframe by...
    1) Loading each seperate patient's dataframe and compute the average baseline severity, normalizing acne severity scores. Then modifies the dataframe, called a severities dataframe.
    2) For each, mapping each value for treatment to the a treatment history tuple. It is a tuple of the form ((days of treatment, ai), (days of treatment, ai+1),....(days of treatment, an))
    where n is the row number of the current day of the particular treatment."""
    
    
    #initializing dict containing severties by day for a given treatment 
    severitiesDayTreatmentDict = {treatment: None for treatment in seperatePatientsDFs[0]["treatment"]}
    modifiedDFs = []
    counter = 0 

    for patient_DF, days_of_intro in zip(seperatePatientsDFs, allPatientsIntroDays):
        #computing average baseline severity for each DF
        average_bl = patient_DF["AcneSeverity"].head(days_of_intro[0][2]).mean()
        #forming new dataframe from old one containing percent severity over baseline 
        modified_DF = patient_DF.copy()
        modified_DF["AcneSeverity"] = modified_DF["AcneSeverity"].apply(lambda x: (x - average_bl)/average_bl)*100
        modifiedDFs.append(modified_DF)
        counter += 1
    
    metadata_DFs = []
    all_patients_treatment_histories = []
    #iterating over all dataframes and their respective treatment days of introduction
    for severities_df, days_of_intro in zip (modifiedDFs, allPatientsIntroDays): 
        #for each dataframe and the corresponding set of days where a given treatment ends
        
        #adding an explicit day column to each severities dataframe to enable indexing with df.loc  
        severities_df["day"] = range(len(severities_df))
    
        #initializing the treatment history metadata column
        severities_df["treatment_history"] = None
    
        last_treat_index = 0 #keeps track of which number of the ordered list of treatments is currently being processed
        treatment_days = {} #keeping track of how many days each particular treatment goes on for
        treatment_history = [] #keeping track of the full history of which treatment occurs before the others and how long they last for
        all_treatment_histories = []
        #also keeping track of the last treatment
        last_treatment_itself = None
        
        #iterating through rows of the dataframe
        for row_index, row in severities_df.iterrows():
            current_day = row["day"]
            current_treatment = row["treatment"]
                   
            #also checking to see if the current treatment is entirely new or falls inside a different treatment block
            if current_treatment != last_treatment_itself:
                treatment_days[current_treatment] = 1  # First day of new treatment
                treatment_history.append((current_treatment, 1))
            else:
                treatment_days[current_treatment] += 1
                treatment_history[-1] = (current_treatment, treatment_days[current_treatment])

            #once history is built, the history is stored in the original dataframe as an entry in own column
            severities_df.at[row_index, "treatment_history"] = list(treatment_history)
            transient_history = treatment_history.copy()
            all_treatment_histories.append(transient_history)
            last_treatment_itself = current_treatment  
            
        #also modifying dataframes to remove baseline acne severities, as they've already been used
        metadata_DFs.append(severities_df[days_of_intro[0][2]:])
        
        all_patients_treatment_histories.append(all_treatment_histories)
    return metadata_DFs, all_patients_treatment_histories

def find_and_plot_severity_states(metadata_DFs):
    """Determines the quantile cutoff determining the quantiles corresponding to categorical acne severity states (low, medium, and high), by
    1) Computing the KDE of the distribution of all normalized acne severity scores over all patients and treatment histories.
    2) Using optimization to find the saddle point of the distribution and consolidating with the modes."""
    fig, axes = plt.subplots(1, figsize=(5, 5))
    # collecting all severities, converting all to positive values, flattening as we go
    all_severities = []
    for df in metadata_DFs:
        all_severities.extend(df["AcneSeverity"] * -1)

    # check for normal character by plotting histogram
    severities_histo = np.histogram(all_severities, density=True)
    # axes.hist(all_severities, bins  = 30, density = True)
    axes.set_title("Acne Severities Distribution (relative to baseline)")
    axes.set_xlabel("Normalized Severity Score")
    axes.set_ylabel("Density")

    # fitting a kernel density estimate to the data
    sns.kdeplot(all_severities, fill=True)
    plt.title("KDE of Acne Severity Distribution")

    # extracting the equation of the pdf and finding the local minimum in between the two modes
    kde_pdf = sp.stats.gaussian_kde(all_severities)

    # using max and min of pdf to find saddle point in between 2 modes, sampling 1000 points
    severity_grid = np.linspace(np.min(all_severities), np.max(all_severities), 1000)
    neg_kde = lambda x: -kde_pdf(x.reshape(1, -1))
    # finding the two main modes using optimization, with first 2 mode guesses at the .2 and .8 quantiles
    guesses = np.percentile(all_severities, [20, 80])

    modes = []
    for guess in guesses:
        better = optimize.minimize(neg_kde, np.array([guess]))
        modes.append(better.x[0])
    modes = np.array(modes)

    # finding the saddle point in between the two modes, using that as cutoff for the two patient states
    initial_guess = np.mean(modes)  # average of the modes
    bds = [(min(modes) + 1, max(modes) - 1)]  # This is a list of two tuples for each mode

    saddle_pt = optimize.minimize(kde_pdf, [initial_guess], bounds=bds)
    state_ranges = [modes[0], saddle_pt.x[0], modes[1]]
    state_names = ["High Severity", "Medium Severity", "Low Severity"]

    # plotting modes and saddle point over the distribution
    plt.scatter(modes[0], kde_pdf(modes[0]), color="red", label="Lower Mode")
    plt.scatter(modes[1], kde_pdf(modes[1]), color="red", label="Upper Mode")
    plt.scatter([saddle_pt.x[0]], kde_pdf([saddle_pt.x[0]]), color='green', label="Saddle Point")

    plt.show()
    plt.close()

    return state_names, state_ranges

def assign_states_to_mdfs(metadata_DFs, state_names, orig_state_ranges):
    """
    MOVED
    Function to construct and attach a second metadata column onto each patient's dataframe. The column contains
    the acne severity categorical state corresponding to a given treatment history."""

    new_ranges = [float(orig_state_ranges[0]), float(orig_state_ranges[1]), float(orig_state_ranges[2])]
    points_in_each_range = defaultdict(list)
    state_and_averages = defaultdict(float)

    for patient_df in metadata_DFs:
        severity_states = np.digitize(-1 * patient_df["AcneSeverity"], new_ranges, right=False)
        severity_states = np.minimum(severity_states, 2)
        patient_df["State"] = [state_names[severity_state] for severity_state in severity_states]

    for patient_df in metadata_DFs:
        states = patient_df["State"].tolist()
        severities = patient_df["AcneSeverity"].tolist()

        for index in range(len(patient_df) - 1):
            state = states[index]
            severity = severities[index]
            points_in_each_range[state].append(-1 * severity)

    for state, points_in_each_range in points_in_each_range.items():
        state_and_averages[state] = np.average(points_in_each_range)

    return metadata_DFs, state_and_averages

def build_histograms(metadata_DFs):
    """This algorithm iterates over all patients' severity dataframes and, for each...

    1) Counts all of the occcurences of each acne severity state throughout all patients, and assigns those counts
    as a value to the treatment history key in the state counts dictionary.
    2) Normalizes the counts into distributions before returning both.
    """

    all_state_counts = defaultdict(Counter)

    for i, patient_df in enumerate(metadata_DFs):
        histories = patient_df["treatment_history"].values
        states = patient_df["State"].values

        for state_index in range(1, len(states)):
            current_state = states[state_index]  # state at position state_index in the patient's dataframe
            current_history = histories[
                state_index]  # metadata (treatment history) at position state_index  in the patient's dataframe
            current_history_key = tuple((str(treatment), int(days)) for treatment, days in current_history)

            # recording the actual counts of severities, with the context of the prior treatment as the key
            all_state_counts[current_history_key][current_state] += 1

    # normalizing counts dictionaries into distributions
    first_order_probabilities = {}

    for previous_treatment, state_counts in all_state_counts.items():
        total_counts = sum(state_counts.values())
        probabilities_given_previous_state = {state: count / total_counts for state, count in state_counts.items()}
        first_order_probabilities[previous_treatment] = probabilities_given_previous_state

    return (all_state_counts, first_order_probabilities)

def build_Dirichlet(prior, all_transition_counts):
    """This function does the following:
    Accepts a prior in order to construct a Bayesian model of each history's distribution as a Dirichlet distribution with a multinomial
    likelihood. It does so by the following methods.
    1)  Iterates through each set of counts for each history and adds them to each parameter in order in the prior, then uses these to
    build the posterior Dirichlet distribution."""

    categories = ['Low Severity', 'Medium Severity', 'High Severity']
    prior_dict = {'Low Severity': prior[0], "Medium Severity": prior[1], "High Severity": prior[2]}

    history_and_posteriors = {}

    for history, count_dict in all_transition_counts.items():
        counts = [count_dict.get(cat, 0) for cat in categories]  # pulling counts from prior dict and counts dict
        prior = [prior_dict[cat] for cat in categories]

        # updating parameters for posterior distribution
        posterior_params = np.array(counts) + np.array(prior)
        # saving posterior params to the dictionary
        history_and_posteriors[history] = posterior_params

    return history_and_posteriors, categories

def data_parsing(data_filename):
    """Function that does the data parsing, given a csv filename in the same directory."""
    raw_data = pd.read_csv(data_filename)
    # actual implementation
    this_seperate_DFs, this_intro_days = seperate_patients(raw_data)
    # this_inspected_data = display_plots_of_dataset(this_seperate_DFs, this_intro_days, .1) #for inspection
    this_md_DFs, this_treatment_history = add_history_metadata(this_seperate_DFs, this_intro_days)
    these_states, these_ranges = find_and_plot_severity_states(this_md_DFs)
    these_assigned_md_DFs, these_state_averages = assign_states_to_mdfs(this_md_DFs, these_states, these_ranges)

    built_histograms, raw_probabilities = build_histograms(these_assigned_md_DFs)
    uninformative_prior = [1, 1, 1]  # with a1 corresponding to low severity, a2 corresponding to medium, and a3 corresponding to high

    these_Dirichlets, these_categories = build_Dirichlet(uninformative_prior, built_histograms)
    return this_treatment_history, these_assigned_md_DFs, these_states, these_ranges, this_md_DFs, these_state_averages, these_Dirichlets