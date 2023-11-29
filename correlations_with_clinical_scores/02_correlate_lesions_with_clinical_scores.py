"""
The script:
 - fetch subjects from the provided /results folders (each one for a specific seed)
 - keep only the unique subjects (to avoid duplicates between seeds)
 - read XLS files (located under /results) with lesion metrics (computed using sct_analyze_lesion separately for GT
 and predicted using our 3D nnUNet model)
 - save the aggregated data to a CSV file
 - create regression plots for each metric (volume, length, max_axial_damage_ratio) manual vs nnUNet and save them to
 the output folder

Note: to read XLS files, you might need to install the following packages:
    pip install openpyxl

Author: Jan Valosek
"""

import os
import re
import glob
import argparse

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import spearmanr

metric_to_title = {'volume': 'Total lesion volume [$mm^3$]',
                   'length': 'Intramedullary lesion length [mm]',
                   'max_axial_damage_ratio': 'Maximal axial damage ratio []'
                   }


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


def format_pvalue(p_value, alpha=0.05, decimal_places=3, include_space=False, include_equal=True):
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
            print(f'WARNING: More than one lesion in {row["fname_lesion_nnunet_3d"]}')
            # Sum the volume and length
            df.at[index, 'volume_'+pred_type] = df_lesion_nnunet_3d['volume [mm3]'].sum()
            df.at[index, 'length_'+pred_type] = df_lesion_nnunet_3d['length [mm]'].sum()
            # Take the max of max_axial_damage_ratio
            df.at[index, 'max_axial_damage_ratio_'+pred_type] = \
                df_lesion_nnunet_3d['max_axial_damage_ratio []'].max()

    return df


def generate_regplot_manual_vs_predicted(df, output_dir):
    """
    Plot data and a linear regression model fit. Manual GT lesion vs lesions predicted using our 3D SCIseg nnUNet model.
    :param df: dataframe with lesion metrics
    :param output_dir: output directory
    """

    for metric in ['volume', 'length', 'max_axial_damage_ratio']:
        # Create a figure
        fig = plt.figure(figsize=(7, 7))
        # Create a subplot
        ax = fig.add_subplot(111)
        # Plot the data (manual vs nnunet_3d) and a linear regression model fit
        # Zurich
        sns.regplot(x=metric+'_manual', y=metric+'_nnunet_3d', data=df[df['site'] == 'zurich'], ax=ax, color='red')
        # Colorado
        sns.regplot(x=metric+'_manual', y=metric+'_nnunet_3d', data=df[df['site'] == 'colorado'], ax=ax, color='blue')
        # Set the title
        ax.set_title(f'{metric_to_title[metric]}: manual vs nnUNet 3D lesion seg')
        # Set the x-axis label
        ax.set_xlabel(f'Manual GT lesion')
        # Set the y-axis label
        ax.set_ylabel(f'Lesion predicted by nnUNet 3D')

        # Make the x- and y-axis limits equal
        ax.set_ylim(ax.get_xlim())
        # Make the x- and y-axis ticks equal
        ax.set_aspect('equal', adjustable='box')

        # Show grid
        ax.grid(True)

        # # Create single custom legend for whole figure with several subplots
        markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in ['red', 'blue']]
        ax.legend(markers, ['Zurich (T2w sag)', 'Colorado (T2w ax)'], numpoints=1, loc='upper left')

        # Save individual figures
        plt.tight_layout()

        # Save the figure
        fname_fig = os.path.join(output_dir, f'{metric}_regplot.png')
        fig.savefig(os.path.join(output_dir, fname_fig), dpi=300)
        print(f'Saved {os.path.join(output_dir, fname_fig)}')
        # Close the figure
        plt.close(fig)


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

    # Compute the difference between initial and discharge scores
    for score in clinical_scores_list:
        df_participants_colorado['diff_' + score] = df_participants_colorado['initial_' + score] - \
                                                    df_participants_colorado['discharge_' + score]

    # Merge the dataframes
    df = pd.merge(df, df_participants_colorado, on='participant_id')

    df_colorado = df[df['site'] == 'colorado']

    # Drop rows with NaNs
    df_colorado = df_colorado.dropna()

    # Compute difference between manual and nnunet_3d lesion segmetation
    for metric in ['volume', 'length', 'max_axial_damage_ratio']:
        df_colorado[metric + '_diff'] = df_colorado[metric + '_manual'] - df_colorado[metric + '_nnunet_3d']

    for metric in ['volume', 'length', 'max_axial_damage_ratio']:
        for score in clinical_scores_list_final + ['diff_' + s for s in clinical_scores_list]:
            # Create a figure
            fig = plt.figure(figsize=(7, 7))
            # Create a subplot
            ax = fig.add_subplot(111)
            # Plot the data (manual vs nnunet_3d) and a linear regression model fit
            # We have clinical scores only for Colorado
            sns.regplot(x=score, y=metric + '_nnunet_3d', data=df_colorado, ax=ax,
                        color='green')
            sns.regplot(x=score, y=metric + '_manual', data=df_colorado, ax=ax,
                        color='orange')
            sns.regplot(x=score, y=metric + '_diff', data=df_colorado, ax=ax,
                        color='black')

            # Compute correlation coefficient and p-value and add them to the plot
            corr_nnunet, pval_nnunet = spearmanr(df_colorado[score], df_colorado[metric + '_nnunet_3d'])
            corr_manual, pval_manual = spearmanr(df_colorado[score], df_colorado[metric + '_manual'])

            # Set the title
            ax.set_title(f'{metric_to_title[metric]} vs {score}')
            # Set the x-axis label
            ax.set_xlabel(f'{score}')
            # Set the y-axis label
            ax.set_ylabel(f'{metric_to_title[metric]}')

            # For AIS, set x-axis ticks to be integers
            if 'ais' in score:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            # Show grid
            ax.grid(True)

            # # Create single custom legend for whole figure with several subplots
            markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in ['green', 'orange', 'black']]
            legend = ax.legend(markers,
                               [f'nnUNet 3D lesion segmentation: Spearman r={corr_nnunet:.2f}, '
                                f'p{format_pvalue(pval_nnunet)}',
                                f'Manual GT lesion segmentation: Spearman r={corr_manual:.2f}, '
                                f'p{format_pvalue(pval_manual)}',
                                f'Difference manual - nnUNet 3D'],
                               numpoints=1,
                               loc='upper right')
            # Make legend's box black
            frame = legend.get_frame()
            frame.set_edgecolor('black')
            ax.add_artist(legend)

            # Save individual figures
            plt.tight_layout()

            # Save the figure
            fname_fig = os.path.join(output_dir, f'{metric}_regplot_{score}.png')
            fig.savefig(os.path.join(output_dir, fname_fig), dpi=300)
            print(f'Saved {os.path.join(output_dir, fname_fig)}')
            # Close the figure
            plt.close(fig)


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

        print(f'Processing XLS files for {row["participant_id"]}')

        # Read the XLS file with lesion metrics for lesion predicted by our 3D SCIseg nnUNet model
        df = fetch_lesion_metrics(index, row, 'nnunet_3d', df)

        # Read the XLS file with lesion metrics for manual (GT) lesion
        df = fetch_lesion_metrics(index, row, 'manual', df)

    # Save the dataframe as XLS file
    df.to_excel(os.path.join(output_dir, 'lesion_metrics.xlsx'), index=False)
    print(f'Saved {os.path.join(output_dir, "lesion_metrics.xlsx")}')

    # Remove rows with volume_manual > 4000
    df = df[df['volume_manual'] <= 4000]

    #  Plot data and a linear regression model fit (manual GT lesion vs lesions predicted using our 3D nnUNet model)
    generate_regplot_manual_vs_predicted(df, output_dir)

    # If sci-colorado participants.tsv file is provided, plot data and a linear regression model fit
    if path_participants_colorado is not None:
        # Plot data and a linear regression model fit. Lesion metrics vs clinical scores.
        generate_regplot_metric_vs_score(df, path_participants_colorado, output_dir)


if __name__ == '__main__':
    main()
