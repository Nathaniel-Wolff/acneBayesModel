#Generated, mostly, with Google Gemini
import pandas as pd
import numpy as np
import argparse


class all_args():
    """CLI for merging datasets. Will be amended later with clinical dataset merging options."""

    def __init__(self, inOpts=None):
        '''
        CommandLine constructor.
        Implements a parser to interpret the command line argv string using argparse.
        '''
        self.parser = argparse.ArgumentParser(
            description="""Pass in name of file to be merged with new synthetic data.""",
            epilog='',
            add_help=True,
            prefix_chars='-',
            usage='%(prog)s [options] -option1[default] <input > output'
        )

        self.parser.add_argument('-o', '--other_data_name', nargs='?', default="sim_acne.csv", action='store',
                                 help='other dataset to merge')
        self.parser.add_argument('-n', '--output_csv_name', nargs='?', default="sim_acne_amended.csv", action='store',
                                 help='name of merged csv file')
        self.parser.add_argument('-m', '--output_diet_csv_name', nargs='?', default="sim_acne_diet.csv", action='store',
                                 help='name of diet data csv file')
        self.parser.add_argument('-in', '--ignore_first_gen', nargs='?', default=True, action='store',
                                 help='use this flag to not create the first dataset again')
        self.parser.add_argument('-im', '--ignore_diet_gen', nargs='?', default=False, action='store',
                                 help='use this flag to not create the first dataset again')

        if inOpts is None:
            self.args = self.parser.parse_args()
        else:
            self.args = self.parser.parse_args(inOpts)

        args = self.parser.parse_args()
def generate_HBM_extra_columns():
    # Setup
    np.random.seed(42)
    n_patients = 10
    days_per_block = 28
    n_blocks = 5
    total_days = days_per_block * n_blocks  # 140
    total_rows = n_patients * total_days

    data = []

    for p in range(n_patients):
        # Establish a patient-specific "baseline" for metabolic traits
        base_insulin = np.random.normal(15, 3)
        base_igf1 = np.random.normal(150, 20)
        base_ph = np.random.normal(5.5, 0.4)  # Normal skin is acidic (~4.7-5.75)

        for day in range(total_days):
            # 1. Determine Treatment Context
            # Blocks: 0:Baseline, 1:Antibiotics, 2:Cream, 3:Antibiotics, 4:Cream
            block_idx = day // days_per_block
            is_antibiotic = 1 if block_idx in [1, 3] else 0
            is_cream = 1 if block_idx in [2, 4] else 0

            # 2. Generate First 5 (Clinical Biomarkers)
            # Add slight random walk/noise to daily levels
            insulin = base_insulin + np.random.normal(0, 0.5)
            igf1 = base_igf1 + np.random.normal(0, 2)

            # Rule: Positive feedback loop between IGF1 and Androgens
            androgen = (igf1 * 0.5) + np.random.normal(10, 1)

            # Rule: Skin pH (affected slightly by cream treatment)
            ph = base_ph + (0.2 if not is_cream else -0.3) + np.random.normal(0, 0.05)

            # Systemic Inflammation (NLR)
            nlr = np.random.normal(2.0, 0.3)

            # 3. Impute Latent Variables (Biological State)
            # Rule: mTORC1 driven by Insulin and IGF1
            mtorc1 = (0.4 * insulin) + (0.1 * igf1) + np.random.normal(5, 1)

            # Rule: Dysbiosis driven by pH (acid-dependent lipase failure) and Androgens
            # Higher pH -> Higher Dysbiosis. Antibiotics -> Lower Dysbiosis.
            dysbiosis_base = (ph * 2.5) + (androgen * 0.05)
            dysbiosis = dysbiosis_base - (5.0 * is_antibiotic) + np.random.normal(0, 1)

            # LKB4 Concentration (Metabolic regulator)
            lkb4 = 100 - (0.5 * mtorc1) + np.random.normal(0, 5)

            data.append([
                p, day, round(insulin, 2), round(igf1, 2), round(ph, 2),
                round(androgen, 2), round(nlr, 2), round(mtorc1, 2),
                round(max(0, dysbiosis), 2), round(lkb4, 2)
            ])

    columns = ['Patient_ID', 'Day_Index', 'Insulin Conc.', 'IGF1 Conc.', 'Skin pH',
               'Bulk Androgen Conc.', 'NLR', 'mTORC1 Conc.', 'Dysbiosis', 'LKB4 Conc.']
    df = pd.DataFrame(data, columns=columns)

    return df
def concat_all_csv(other_csv_filename, other_df, out_name):
    old_synthetic_df = pd.read_csv(other_csv_filename)
    merged_df = pd.concat([old_synthetic_df, other_df], axis=1)
    merged_df.to_csv(out_name, index=False)

    return None
def generate_HBM_additional_data(out_name, num_patients = 10, num_blocks = 5, days_per_block = 28):
    """Function to create synthetic data for NLME of treamtment response and diet. Uses the following column headers:
    Day -- ClPhos-BPO (g) -- Isotret (g) -- Fiber (g) -- Carbs (g) -- Protein (g) -- Sat Fat (g) --Unsat Fat (g) -- Water (mL) -- Sodium (g) -- iAUC
    For three meals a day, and 1-2 applications of treatments. Clinicians can add more applications/change dosages in the UI.
    5 treatment blocks of 28 days each are generated here. Each treatment block gradually increases the dosages by 25%, and
    application frequency increases from 1 to 2 times per day after the third treatment block. """

    columns = ["Patient_ID", "Day", "ClPhos-BPO (g)", "Isotret (g)", "Fiber (g)", "Carbs (g)", "Protein (g)", "Sat Fat (g)", "Unsat Fat (g)", "Water (mL)",
               "Sodium (g)", "iAUC"]

    #setup
    np.random.seed(42)
    number_days = num_blocks * days_per_block
    base_dose_ClPhos_BPO_topical = np.random.normal(0.3, 0.2) #about a pea sized amount
    base_dose_isotret_oral = 60.0 * np.random.normal(0.000375, 0.000012)  #assuming accutane usage here, assuming average weight of subjects is around 60 kg
    optimal_added_osmolarity = 2.3 / 2.4 #hardcoded for now (g/L)

    all_data = []
    for p in range(num_patients):
        #hierarhical priors for each intake
        p_mean_fiber = np.random.normal(10.0, 9.0)  # assuming normalized mean intake of 10 and stdev of about 3
        p_mean_carbs = np.random.normal(35.0, 21.0)  # assuming mean intake of 35 and stdev of about 7
        p_mean_protein = np.random.normal(20.0, 10.0)  # assuming mean intake of 20 and stdev of 10
        p_mean_sat_fat = np.random.normal(7.0, 4.0)  # assuming mean intake of 7 and stdev of 3
        p_mean_unsat_fat = np.random.normal(25.0, 13.0)  # assuming mean intake of 25 and stdev of 9
        p_mean_sodium = np.random.normal(1.1, .6)  # assuming mean intake of 1.1 g and stdev of 9
        p_mean_water = np.random.normal(100, 20)  # assuming mean intake of 250 ml and stdev of 9

        #weighted average for mean of iAUC distribution

        total_carbs = p_mean_fiber + p_mean_carbs
        total_fats = p_mean_sat_fat + p_mean_unsat_fat
        total_macros = total_carbs + total_fats + p_mean_protein
        total_electrolyte_g_l = p_mean_sodium / (1000 * p_mean_water)
        # assuming full dispersion in body fluids throughout the day, upsetting osmotic balance or not
        electrolytes_gain = total_electrolyte_g_l/ optimal_added_osmolarity
        macros_ratios = np.array([p_mean_carbs/total_carbs, -p_mean_protein/total_macros, total_fats/total_macros, electrolytes_gain]          )
        averaged = np.average(a = macros_ratios, weights = [0.6, 0.1, 0.25, .05]) #initial guess
        p_mean_iAUC = np.random.normal(averaged, 0.1)

        for day in range(number_days):
            day_index = day // days_per_block

            for meal_index in range(3): #also using to set treatment
                fiber = np.random.normal( (1 + .2 * meal_index) * p_mean_fiber , 2.0) #fiber increases somewhat throughout the day
                carbs = np.random.normal(p_mean_carbs, 2.0) if meal_index > 0 else np.random.normal(p_mean_carbs * 1.4, 2.0)#breakfasts tend to be carb heavy
                protein = np.random.normal(p_mean_protein, 2.0)
                sat_fat = np.random.normal(p_mean_sat_fat, 2.0)
                unsat_fat = np.random.normal(p_mean_unsat_fat, 2.0)
                sodium = np.random.normal(p_mean_sodium * (1 + .1 * meal_index), .1) #typically less salt early in the day, more later from unhealthy snacks
                water = np.random.normal(p_mean_water, 10)
                iAUC = np.abs(np.random.normal(p_mean_iAUC, .3))#decently high stdev, enforcing positivity

                clin_dose = base_dose_ClPhos_BPO_topical * np.power(1.25, day_index) if meal_index == 0 or meal_index == 2 else 0
                isotret_dose = base_dose_isotret_oral * np.power(1.25, day_index) if meal_index == 0 or meal_index == 2 else 0

                all_data.append([
                    p, day, round(clin_dose, 4), round(isotret_dose, 4), round(fiber, 4),
                    round(carbs, 4), round(protein, 4), round(sat_fat, 4),
                    round(unsat_fat, 4), round(sodium, 2), round(water, 2), round(iAUC, 4)
                ])
    frame = pd.DataFrame(all_data, columns=columns)
    frame.to_csv(out_name, index=False)
    return frame

def main(out_name=None, outFile=None, options=None):
    cli = all_args(options)
    other_dataset_name = cli.args.other_data_name
    other_outfile_name  = cli.args.output_csv_name
    diet_dataset_name = cli.args.output_diet_csv_name
    ignore_first = cli.args.ignore_first_gen
    ignore_second = cli.args.ignore_diet_gen

    if not ignore_first:
        actual_other_df = generate_HBM_extra_columns()
        concatenated = concat_all_csv(other_dataset_name, actual_other_df, other_outfile_name)

    if not ignore_second:
        diet_df = generate_HBM_additional_data(diet_dataset_name)


if __name__ == "__main__":
    main()

