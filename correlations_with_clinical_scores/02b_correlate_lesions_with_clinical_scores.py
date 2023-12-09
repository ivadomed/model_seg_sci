"""
The script:
 - fetch subjects from the provided /results folders (each one for a specific seed)
 - keep only the unique subjects (to avoid duplicates between seeds)
 - read XLS files (located under /results) with lesion metrics (computed using sct_analyze_lesion separately for GT
 and predicted using our 3D nnUNet model)
 - save the aggregated data to a CSV file
 - plot data and a linear regression model fit for each each metric (volume, length, max_axial_damage_ratio) manual vs
 nnUNet lesion segmentation and save them to the output folder
 - plot data and a linear regression model fit for each each metric (volume, length, max_axial_damage_ratio) manual vs
 clinical scores (LEMS, AIS, pinprick, light touch) and admission (initial), discharge and difference and save them to
 the output folder

Note: to read XLS files, you might need to install the following packages:
    pip install openpyxl

Author: Jan Valosek
"""

import os
import sys
import re
import glob
import argparse
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import spearmanr, wilcoxon

FONT_SIZE = 18

metric_to_title = {'volume': 'Lesion Volume',
                   'length': 'Intramedullary Lesion Length',
                   'max_axial_damage_ratio': 'Maximal Axial Damage Ratio'
                   }

score_to_title = {'initial_LEMS': 'Initial Lower Extremity Motor Score (LEMS)',
                  'initial_ais': 'Initial ASIA Impairment Scale (AIS)',
                  'initial_ais_grouped': 'Initial ASIA Impairment Scale (AIS)',
                  'initial_pin_prick_total': 'Initial Pin Prick Score',
                  'initial_light_touch_total': 'Initial Light Touch Score',
                  'discharge_LEMS': 'Discharge Lower Extremity Motor Score (LEMS)',
                  'discharge_ais': 'Discharge ASIA Impairment Scale (AIS)',
                  'discharge_ais_grouped': 'Discharge ASIA Impairment Scale (AIS)',
                  'discharge_pin_prick_total': 'Discharge Pin Prick Score',
                  'discharge_light_touch_total': 'Discharge Light Touch Score',
                  'diff_LEMS': 'Difference (Discharge-Initial) Lower Extremity Motor Score (LEMS)',
                  'diff_ais': 'Difference (Discharge-Initial) ASIA Impairment Scale (AIS)',
                  'diff_pin_prick_total': 'Difference (Discharge-Initial) Pin Prick Score',
                  'diff_light_touch_total': 'Difference (Discharge-Initial) Light Touch Score'
                  }

metric_to_label = {'volume': 'Total Lesion Volume [mm$^3$]',
                   'length': 'Intramedullary Lesion Length [mm]',
                   'max_axial_damage_ratio': 'Maximal Axial Damage Ratio'
                   }

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='TODO',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-i',
        required=True,
        nargs='+',
        help='Space separated list of paths to the \'results\' folders with XLS files. Each \'results\' folder '
             'corresponds to a specific seed. The results folders were generated using the \'01_analyze_lesions.sh\'.'
             'The XLS files contain lesion metrics and were generated using \'sct_analyze_lesion.\' '
             'Example: sci-multisite_analyze_lesions_seed7/results sci-multisite_analyze_lesions_seed42/results'
    )
    parser.add_argument(
        '-participants-tsv-colorado',
        required=False,
        help='Full path to the sci-colorado participants.tsv file containing the clinical scores.'
    )
    parser.add_argument(
        '-o',
        required=False,
        default='stats',
        help='Path to the output folder where stats and figures will be saved. Default: ./stats'
    )

    return parser


def fetch_subject_and_session(filename_path):
    """
    Get subject ID, session ID and filename from the input BIDS-compatible filename or file path
    The function works both on absolute file path as well as filename
    :param filename_path: input nifti filename (e.g., sub-001_ses-01_T1w.nii.gz) or file path
    (e.g., /home/user/MRI/bids/derivatives/labels/sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz
    :return: subject_session: subject ID and session ID (e.g., sub-001_ses-01) or subject ID (e.g., sub-001)
    """

    subject = re.search('sub-(.*?)[_/]', filename_path)     # [_/] means either underscore or slash
    subjectID = subject.group(0)[:-1] if subject else ""    # [:-1] removes the last underscore or slash

    session = re.search('ses-(.*?)[_/]', filename_path)     # [_/] means either underscore or slash
    sessionID = session.group(0)[:-1] if session else ""    # [:-1] removes the last underscore or slash
    # REGEX explanation
    # . - match any character (except newline)
    # *? - match the previous element as few times as possible (zero or more times)

    subject_session = subjectID + '_' + sessionID if subjectID and sessionID else subjectID

    return subject_session


def format_pvalue(p_value, alpha=0.05, decimal_places=3, include_space=True, include_equal=True):
    """
    Format p-value.
    If the p-value is lower than alpha, format it to "<0.001", otherwise, round it to three decimals

    :param p_value: input p-value as a float
    :param alpha: significance level
    :param decimal_places: number of decimal places the p-value will be rounded
    :param include_space: include space or not (e.g., ' = 0.06')
    :param include_equal: include equal sign ('=') to the p-value (e.g., '=0.06') or not (e.g., '0.06')
    :return: p_value: the formatted p-value (e.g., '<0.05') as a str
    """
    if include_space:
        space = ' '
    else:
        space = ''

    # If the p-value is lower than alpha, return '<alpha' (e.g., <0.001)
    if p_value < alpha:
        p_value = space + "<" + space + str(alpha)
    # If the p-value is greater than alpha, round it number of decimals specified by decimal_places
    else:
        if include_equal:
            p_value = space + '=' + space + str(round(p_value, decimal_places))
        else:
            p_value = space + str(round(p_value, decimal_places))

    return p_value


def get_fnames(dir_paths):
    """
    Get the list of file names across all input paths (i.e. all seeds), sort them and keep only the unique ones
    :param dir_paths: list of paths to the 'results' folders containing the XLS files with lesion metrics for each seed
    :return: pandas dataframe with the paths to the XLS files for manual and predicted lesions
    """
    # Initialize an empty list file names across all input paths (i.e. all seeds)
    fname_files_all = list()

    # Get all file names across all input paths (i.e. all seeds)
    for dir_path in dir_paths:
        # Get lesion metrics for lesions predicted by our 3D SCIseg nnUNet model (_lesion_nnunet_3d_analysis.xls)
        fname_files = glob.glob(os.path.join(dir_path, '**', '*_lesion_nnunet_3d_analysis.xls'), recursive=True)
        # if fname_files is empty, exit
        if len(fname_files) == 0:
            print(f'ERROR: No _lesion_nnunet_3d_analysis.xls files found in {dir_path}')

        fname_files_all.extend(fname_files)

    # Sort the list of file names (to make the list the same when provided the input folders in different order)
    fname_files_all.sort()

    # Convert fname_files_all into pandas dataframe
    df = pd.DataFrame(fname_files_all, columns=['fname_lesion_nnunet_3d'])

    # Add a column with participant_id
    df['participant_id'] = df['fname_lesion_nnunet_3d'].apply(lambda x: fetch_subject_and_session(x))
    # Make the participant_id column the first column
    df = df[['participant_id', 'fname_lesion_nnunet_3d']]
    # Add a column with site (if participant_id contains 'zh' --> site = 'zurich', otherwise site = 'colorado')
    df['site'] = df['participant_id'].apply(lambda x: 'zurich' if 'zh' in x else 'colorado')

    print(f'Number of rows: {len(df)}')

    # Keep only unique participant_id rows
    df = df.drop_duplicates(subset=['participant_id'])
    print(f'Number of unique participants: {len(df)}')

    # Add a column with fname_lesion_manual by replacing '_nnunet_3d' by '-manual_bin'
    df['fname_lesion_manual'] = df['fname_lesion_nnunet_3d'].apply(lambda x: x.replace('_nnunet_3d', '-manual_bin'))

    # Make sure the lesion exists
    for fname in df['fname_lesion_manual']:
        if not os.path.exists(fname):
            raise ValueError(f'ERROR: {fname} does not exist.')

    return df


def fetch_lesion_metrics(index, row, pred_type, df):
    """
    Fetch lesion metrics from the XLS file with lesion metrics generated by sct_analyze_lesion
    :param index: index of the dataframe
    :param row: row of the dataframe (one row corresponds to one participant)
    :param pred_type: nnunet_3d or manual
    :param df: dataframe with the paths to the XLS files for manual and predicted lesions
    :return: df: updated dataframe with the paths to the XLS files for manual and predicted lesions and lesion metrics
    """
    # Read the XLS file with lesion metrics for lesion predicted by our 3D SCIseg nnUNet model
    df_lesion_nnunet_3d = pd.read_excel(row['fname_lesion_'+pred_type], sheet_name='measures')
    # If the dataframe is empty (there is no lesion), insert nan values
    if df_lesion_nnunet_3d.empty:
        df.at[index, 'volume_'+pred_type] = np.nan
        df.at[index, 'length_'+pred_type] = np.nan
        df.at[index, 'max_axial_damage_ratio_'+pred_type] = np.nan
    else:
        # One lesion
        if len(df_lesion_nnunet_3d) == 1:
            # Get volume, length, and max_axial_damage_ratio and save the values in the currently processed df row
            df.at[index, 'volume_'+pred_type] = df_lesion_nnunet_3d['volume [mm3]'].values[0]
            df.at[index, 'length_'+pred_type] = df_lesion_nnunet_3d['length [mm]'].values[0]
            df.at[index, 'max_axial_damage_ratio_'+pred_type] = \
                df_lesion_nnunet_3d['max_axial_damage_ratio []'].values[0]
        # More than one lesion
        else:
            logger.info(f'WARNING: More than one lesion in {row["fname_lesion_nnunet_3d"]}')
            # Sum the volume and length
            df.at[index, 'volume_'+pred_type] = df_lesion_nnunet_3d['volume [mm3]'].sum()
            df.at[index, 'length_'+pred_type] = df_lesion_nnunet_3d['length [mm]'].sum()
            # Take the max of max_axial_damage_ratio
            df.at[index, 'max_axial_damage_ratio_'+pred_type] = \
                df_lesion_nnunet_3d['max_axial_damage_ratio []'].max()

    return df


def plot_everything(df_colorado, clinical_scores_list, clinical_scores_list_final, output_dir):
    """
    Plot everything on initial, discharge, and diff clinical scores
    Manual GT vs SCIseg nnUNet 3D predictions
    :param df_colorado:
    :param clinical_scores_list:
    :param clinical_scores_list_final:
    :param output_dir:
    :return:
    """

    mpl.rcParams['font.family'] = 'Helvetica'

    # Loop across lesion metrics
    for metric in ['volume', 'length', 'max_axial_damage_ratio']:

        # Compute Wilcoxon signed-rank test between manual and nnunet_3d
        stat, pval = wilcoxon(df_colorado[metric + '_manual'], df_colorado[metric + '_nnunet_3d'])
        logger.info(f'{metric}: p{format_pvalue(pval, alpha=0.001)}')

        # Loop across clinical scores
        for score in clinical_scores_list_final + ['diff_' + s for s in clinical_scores_list] + \
                     ['initial_ais_grouped', 'discharge_ais_grouped']:
            # Create a figure
            fig = plt.figure(figsize=(6, 6))
            # Create a subplot
            ax = fig.add_subplot(111)
            # Plot the data (manual vs nnunet_3d) and a linear regression model fit
            sns.regplot(x=score, y=metric + '_nnunet_3d', data=df_colorado, ax=ax,
                        color='green')
            sns.regplot(x=score, y=metric + '_manual', data=df_colorado, ax=ax,
                        color='orange')
            # sns.regplot(x=score, y=metric + '_diff', data=df_colorado, ax=ax,
            #             color='black')

            # Compute correlation coefficient and p-value and add them to the plot
            corr_nnunet, pval_nnunet = spearmanr(df_colorado[score], df_colorado[metric + '_nnunet_3d'])
            corr_manual, pval_manual = spearmanr(df_colorado[score], df_colorado[metric + '_manual'])

            # Set the title
            #ax.set_title(f'{metric_to_title[metric]} vs {score_to_title[score]}', fontsize=FONT_SIZE)
            # Set the x-axis label
            ax.set_xlabel(f'{score_to_title[score]}', fontsize=FONT_SIZE)
            # Set the y-axis label
            ax.set_ylabel(f'{metric_to_label[metric]}', fontsize=FONT_SIZE)

            # For AIS, set x-axis ticks to be integers
            if 'ais' in score:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            # For grouped AIS, change x-axis labels from 1 to motor complete (A/B) and from 2 to motor incomplete (C/D)
            if 'ais_grouped' in score:
                ax.set_xticklabels(['', 'motor complete (A/B)', 'motor incomplete (C/D)'])

            # Show grid
            ax.grid(True)

            # # Create single custom legend for whole figure with several subplots
            # markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in ['green', 'orange', 'black']]
            # legend = ax.legend(markers,
            #                    [f'SCISeg 3D Prediction: Spearman r={corr_nnunet:.2f}, '
            #                     f'p{format_pvalue(pval_nnunet)}',
            #                     f'Manual Ground Truth: Spearman r={corr_manual:.2f}, '
            #                     f'p{format_pvalue(pval_manual)}',
            #                     f'Difference manual - nnUNet 3D'],
            #                    numpoints=1,
            #                    loc='upper right')
            # Create single custom legend for whole figure with several subplots
            markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in ['orange', 'green']]
            legend = ax.legend(markers,
                               [f'Manual Ground Truth: r = {corr_manual:.2f}, '
                                f'p{format_pvalue(pval_manual)}',
                                f'SCISeg 3D Prediction: r = {corr_nnunet:.2f}, '
                                f'p{format_pvalue(pval_nnunet)}'],
                               numpoints=1,
                               loc='upper right', fontsize=FONT_SIZE-4)

            # Adjust y-axis limits
            if metric == 'length':
                ax.set_ylim(-5, 120)
            elif metric == 'volume':
                ax.set_ylim(-200, 3000)
            else:
                ax.set_ylim(-0.1, 1)

            # Make legend's box black
            frame = legend.get_frame()
            frame.set_edgecolor('black')
            ax.add_artist(legend)

            # Increase tick labels size
            ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 3)

            # Save individual figures
            plt.tight_layout()

            # Save the figure
            fname_fig = os.path.join(output_dir, f'{metric}_regplot_{score}.png')
            fig.savefig(os.path.join(output_dir, fname_fig), dpi=300)
            print(f'Saved {os.path.join(output_dir, fname_fig)}')
            # Close the figure
            plt.close(fig)


def plot_improvers_vs_nonimprovers_discharge(df_colorado, clinical_scores_list, output_dir):
    """
    Plot improvers vs non-improvers on discharge clinical scores
    :param df_colorado:
    :param clinical_scores_list:
    :param output_dir:
    :return:
    """

    mpl.rcParams['font.family'] = 'Helvetica'

    # Loop across lesion metrics
    for metric in ['volume', 'length', 'max_axial_damage_ratio']:
        # Loop across discharge clinical scores
        for score in ['discharge_' + s for s in clinical_scores_list]:
        #for score in ['diff_' + s for s in clinical_scores_list]:

            # skip ais for # OPTION #3
            if 'ais' in score:
                continue

            # Plot the data (SCISeg 3D) and a linear regression model fit
            sns.lmplot(x=score, y=metric + '_nnunet_3d', data=df_colorado, palette="Set1",
                       #hue=score + '_improve', legend=False)       # OPTION #1
                       #hue='improve', legend=False)                # OPTION #2
                       hue=score.replace('discharge', 'improve'), legend=False)        # OPTION #3
                       #hue = score.replace('diff', 'improve'), legend = False)  # OPTION #3

            # Change the figure size
            plt.gcf().set_size_inches(7, 7)

            # Set the title
            plt.title(f'{metric_to_title[metric]} vs {score}')
            # Set the x-axis label
            plt.xlabel(f'{score}')
            # Set the y-axis label
            plt.ylabel(f'{metric_to_title[metric]} (computed from SCISeg 3D prediction)')

            # For AIS, set x-axis ticks to be integers
            if 'ais' in score:
                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

            # Show grid
            plt.grid(True)

            # Move legend to the upper right corner
            plt.legend(loc='upper right')
            # Add legened title
            plt.gca().get_legend().set_title('Improvement')

            # Save individual figures
            plt.tight_layout()

            # Save the figure
            fname_fig = os.path.join(output_dir, f'{metric}_lmplot_{score}.png')
            plt.savefig(os.path.join(output_dir, fname_fig), dpi=300)
            print(f'Saved {os.path.join(output_dir, fname_fig)}')
            # Close the figure
            plt.close()


def generate_regplot_metric_vs_score(df, path_participants_colorado, output_dir):
    """
    Plot data and a linear regression model fit. Lesion metrics vs clinical scores.
    :param df: dataframe with lesion metrics
    :param path_participants_colorado: path to the participants.tsv file
    :param output_dir: output directory
    """

    # List of clinical scores
    clinical_scores_list = ['LEMS', 'ais', 'pin_prick_total', 'light_touch_total']

    clinical_scores_list_final = []
    # Each score is available in the initial and discharge versions
    for score in clinical_scores_list:
        clinical_scores_list_final.append('initial_' + score)
        clinical_scores_list_final.append('discharge_' + score)

    # Read the participants.tsv file (participant_id and clinical scores columns)
    df_participants_colorado = pd.read_csv(path_participants_colorado, sep='\t',
                                           usecols=['participant_id'] + clinical_scores_list_final)

    # Recode discharge_ais
    df_participants_colorado['discharge_ais'] = df_participants_colorado['discharge_ais'].replace(
        {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5})

    # Separate initial_ais into 2 groups of motor complete (AIS A/B) versus motor incomplete (AIS C/D)
    df_participants_colorado['initial_ais_grouped'] = df_participants_colorado['initial_ais'].replace(
        {1: 1, 2: 1, 3: 2, 4: 2})

    # Separate discharge_ais into 2 groups of motor complete (AIS A/B) versus motor incomplete (AIS C/D)
    df_participants_colorado['discharge_ais_grouped'] = df_participants_colorado['discharge_ais'].replace(
        {1: 1, 2: 1, 3: 2, 4: 2})

    # Compute the difference between initial and discharge scores
    for score in clinical_scores_list:
        df_participants_colorado['diff_' + score] = df_participants_colorado['discharge_' + score] - \
                                                    df_participants_colorado['initial_' + score]

    # OPTION #1
    # # Define 2 groups of "improvers" versus "non-improvers": "improvers" (1) with diff_SCORE >= 1 and
    # # "non-improvers" (0) with diff_SCORE < 1
    # for score in clinical_scores_list:
    #     df_participants_colorado['diff_' + score + '_improve'] = np.where(
    #         df_participants_colorado['diff_' + score] >= 1, 1, 0)

    # OPTION #2
    # Define 2 groups of "improvers" versus "non-improvers": "improvers" (1) with discharge_ais_grouped >= 1 and
    # "non-improvers" (0) with discharge_ais_grouped == 0
    # df_participants_colorado['improve'] = np.where(df_participants_colorado['diff_ais'] >= 1, 0, 1)

    # OPTION #3
    # Define 2 groups of "improvers" versus "non-improvers" using the minimal detectable change for LEMS, light touch,
    # and pin prick. LEMS = 1 point, pin prick = 7.8 points, light touch = 12.95 points
    # Details: https://www.sralab.org/rehabilitation-measures/international-standards-neurological-classification-spinal-cord-injury-asia-impairment-scale
    df_participants_colorado['improve_LEMS'] = np.where(df_participants_colorado['diff_LEMS'] >= 1, 1, 0)
    df_participants_colorado['improve_light_touch_total'] = np.where(
        df_participants_colorado['diff_light_touch_total'] >= 12.95, 1, 0)
    df_participants_colorado['improve_pin_prick_total'] = np.where(
        df_participants_colorado['diff_pin_prick_total'] >= 7.8, 1, 0)

    # Merge the dataframes
    df = pd.merge(df, df_participants_colorado, on='participant_id')

    # Keep only Colorado - we have clinical scores only for Colorado
    df_colorado = df[df['site'] == 'colorado']

    # Drop rows with NaNs
    df_colorado = df_colorado.dropna()

    # Compute difference between manual and nnunet_3d lesion segmentation
    for metric in ['volume', 'length', 'max_axial_damage_ratio']:
        df_colorado[metric + '_diff'] = df_colorado[metric + '_manual'] - df_colorado[metric + '_nnunet_3d']

    # Initial, discharge, and difference scores
    plot_everything(df_colorado, clinical_scores_list, clinical_scores_list_final, output_dir)
    # Improvers vs non-improvers on discharge scores only
    plot_improvers_vs_nonimprovers_discharge(df_colorado, clinical_scores_list, output_dir)


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Path to the sci-colorado participants.tsv file containing the clinical scores
    path_participants_colorado = args.participants_tsv_colorado

    # Output directory
    output_dir = os.path.join(os.getcwd(), args.o)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Created {output_dir}')

    # From output_dir, get '2sites', '3sites_beforeAL', or '3sites_afterAL' using regex
    match = re.search(r'2sites|3sites_beforeAL|3sites_afterAL', output_dir)
    if match:
        figure_title = match.group()

    # Dump log file there
    fname_log = f'log_{figure_title}.txt'
    if os.path.exists(fname_log):
        os.remove(fname_log)
    fh = logging.FileHandler(os.path.join(os.path.abspath(output_dir), fname_log))
    logging.root.addHandler(fh)

    # Parse input paths
    dir_paths = [os.path.join(os.getcwd(), path) for path in args.i]

    # Check if the input path exists
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            raise ValueError(f'ERROR: {dir_path} does not exist.')

    # For each participant_id, get the lesion and spinal cord file names
    df = get_fnames(dir_paths)

    # Iterate over the rows of the dataframe and read the XLS files
    for index, row in df.iterrows():

        logger.info(f'Processing XLS files for {row["participant_id"]}')

        # Read the XLS file with lesion metrics for lesion predicted by our 3D SCIseg nnUNet model
        df = fetch_lesion_metrics(index, row, 'nnunet_3d', df)

        # Read the XLS file with lesion metrics for manual (GT) lesion
        df = fetch_lesion_metrics(index, row, 'manual', df)

    # Save the dataframe as XLS file
    xls_fname = os.path.join(output_dir, f'lesion_metrics_{figure_title}.xlsx')
    df.to_excel(xls_fname, index=False)
    logger.info(f'Saved {xls_fname}')

    # Replace nan with 0 for the volume_nnunet_3d column
    # nan means that there is no lesion predicted by our 3D nnUNet model;
    # therefore, metrics are assigned to 0 to plot them
    df['volume_nnunet_3d'] = df['volume_nnunet_3d'].fillna(0)
    df['length_nnunet_3d'] = df['length_nnunet_3d'].fillna(0)
    df['max_axial_damage_ratio_nnunet_3d'] = df['max_axial_damage_ratio_nnunet_3d'].fillna(0)

    # Remove rows with volume_manual > 4000
    #df = df[df['volume_manual'] <= 4000]

    # If sci-colorado participants.tsv file is provided, plot data and a linear regression model fit
    if path_participants_colorado is not None:
        # Plot data and a linear regression model fit. Lesion metrics vs clinical scores.
        generate_regplot_metric_vs_score(df, path_participants_colorado, output_dir)


if __name__ == '__main__':
    main()
