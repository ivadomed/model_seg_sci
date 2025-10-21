"""
Script to read 1) CSV file with automatically measured lesion metrics (computed using SCIsegV2 + sct_analyze_lesion)
and 2) XLSX file with manually measured lesion metrics and clinical scores. Merge both files and normalize sensorimotor
scores based on time since injury. Save the merged dataframe as a CSV file.

Author: Jan Valosek
"""

import os
import argparse
import pandas as pd

from utils import read_file_sct, read_file_manual_sci_zurich, read_file_manual_nisci_trial, normalize_sensorimotor_scores


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='Read 1) CSV file with automatically measured lesion metrics (computed using SCIsegV2 + sct_analyze_lesion) '
                    'and 2) XLSX file with manually measured lesion metrics and clinical scores. Merge both files and '
                    'normalize sensorimotor scores based on time since injury. Save the merged dataframe as a CSV file.',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-file-sct',
        required=True,
        type=str,
        help='Absolute path to a CSV file with lesion metrics computed using sct_analyze_lesion. '
             'The file should contain metrics aggregated across subjects. You can generate this file using the '
             '02_combine_xlsx_files.py script. '
    )
    parser.add_argument(
        '-file-manual',
        required=True,
        type=str,
        help='Absolute path to an XLSX file with manually measured lesion metrics and clinical scores. '
    )

    return parser


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Read the data
    file_sct = args.file_sct
    file_manual = args.file_manual

    #----------------
    # CSV file with lesion metrics computed using sct_analyze_lesion
    #----------------
    df_sct = read_file_sct(file_sct)

    #----------------
    # XLSX file with manually measured lesion metrics and clinical scores
    #----------------
    if 'nisci' in file_manual.lower():
        df_manual = read_file_manual_nisci_trial(file_manual)
    elif 'sci-zurich' in file_manual.lower():
        df_manual = read_file_manual_sci_zurich(file_manual)

    #----------------
    # Merge the dataframes
    #----------------
    if 'nisci' in file_manual.lower():
        # Keep only 'ses-01' session from df_sct
        df_sct = df_sct[df_sct['session_id'] == 'ses-01']
        # Drop the 'session_id' column
        df_sct = df_sct.drop(columns=['session_id'])
        df = pd.merge(df_sct, df_manual, on=['participant_id'])
        # Reorder columns to 'participant_id', 'mri_time_since_injury', ...
        cols = df.columns.tolist()
        cols.remove('participant_id')
        cols.remove('mri_time_since_injury')
        cols = ['participant_id', 'mri_time_since_injury'] + cols
        df = df[cols]
    elif 'sci-zurich' in file_manual.lower():
        df = pd.merge(df_sct, df_manual, on=['participant_id', 'session_id'])
        # Reorder columns to 'participant_id', 'session_id', 'mri_time_since_injury', ...
        cols = df.columns.tolist()
        cols.remove('participant_id')
        cols.remove('session_id')
        cols.remove('mri_time_since_injury')
        cols = ['participant_id', 'session_id', 'mri_time_since_injury'] + cols
        df = df[cols]


    #----------------
    # Normalize sensorimotor scores
    #----------------
    print(f'Number of subjects: {df.shape[0]}')

    if 'nisci' in file_manual.lower():
        time_points = ['01', '02', '03', '04', '05', '06']
    elif 'sci-zurich' in file_manual.lower():
        time_points = ['bl', '1m', '3m', '6m', '12m']

    df = normalize_sensorimotor_scores(df, time_points=time_points)

    # Save df as CSV
    output_dir = os.path.dirname(file_manual)
    merged_csv_fname = os.path.join(output_dir, f'lesion_metrics_and_normalized_clinical_scores.csv')
    df.to_csv(merged_csv_fname, index=False)
    print(f'Dataframe saved as {merged_csv_fname}')


if __name__ == '__main__':
    main()
