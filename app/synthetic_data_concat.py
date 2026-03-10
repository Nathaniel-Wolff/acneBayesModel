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

    # Output first few rows to verify
    #print(df.head(15))

    return df

def concat_all_csv(other_csv_filename, other_df, out_name):
    old_synthetic_df = pd.read_csv(other_csv_filename)
    merged_df = pd.concat([old_synthetic_df, other_df], axis=1)
    merged_df.to_csv(out_name, index=False)

    return None


def main(out_name=None, outFile=None, options=None):
    cli = all_args(options)
    other_dataset_name = cli.args.other_data_name
    other_outfile_name  = cli.args.output_csv_name
    actual_other_df = generate_HBM_extra_columns()
    concatenated = concat_all_csv(other_dataset_name, actual_other_df, other_outfile_name)

if __name__ == "__main__":
    main()

