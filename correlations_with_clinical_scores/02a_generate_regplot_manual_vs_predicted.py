"""
Generate sns.regplot for each metric (volume, length, max_axial_damage_ratio) manual vs SCIseg 3D lesion segmentation
BEFORE (phase1) and AFTER (phase3) active learning.

The script:
 - fetch subjects from the provided /results folders (each one for a specific seed)
 - keep only the unique subjects (to avoid duplicates between seeds)
 - read XLS files (located under /results) with lesion metrics (computed using sct_analyze_lesion separately for GT
 and predicted using our 3D nnUNet model)
 - plot data and a linear regression model fit for each metric (volume, length, max_axial_damage_ratio) manual vs
 nnUNet lesion segmentation and save them to the output folder

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
from scipy.stats import wilcoxon

metric_to_title = {'volume': 'Total Lesion Volume',
                   'length': 'Intramedullary Lesion Length',
                   'max_axial_damage_ratio': 'Maximal Axial Damage Ratio'
                   }

metric_to_axis = {'volume': '[mm$^3$]',
                   'length': '[mm]',
                   'max_axial_damage_ratio': ''
                  }

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)

FONT_SIZE = 15


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='Plot MRI manual vs SCIseg 3D lesion segmentation metrics BEFORE (phase1) and AFTER (phase3) '
                    'active learning',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-method1',
        required=True,
        nargs='+',
        help='BeforeAL (phase1): Space separated list of paths to the \'results\' folders with XLS files. '
             'Each \'results\' folder corresponds to a specific seed. The results folders were generated using the '
             '\'01_analyze_lesions.sh\'.'
             'The XLS files contain lesion metrics and were generated using \'sct_analyze_lesion.\' '
             'Example: clinical_correlation_2sites/seed7/results '
             'clinical_correlation_2sites/seed42/results ...'
    )
    parser.add_argument(
        '-method2',
        required=True,
        nargs='+',
        help='AfterAL (phase3): Space separated list of paths to the \'results\' folders with XLS files. '
             'Each \'results\' folder corresponds to a specific seed. The results folders were generated using the '
             '\'01_analyze_lesions.sh\'.'
             'The XLS files contain lesion metrics and were generated using \'sct_analyze_lesion.\' '
             'Example: clinical_correlation_3sites_afterAL/seed7/results '
             'clinical_correlation_3sites_afterAL/seed42/results ...'
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

    # Check if the XLS file with lesion metrics for manual lesion exists
    if not os.path.exists(row['fname_lesion_'+pred_type]):
        raise ValueError(f'ERROR: {row["fname_lesion_"+pred_type]} does not exist.')

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

    mpl.rcParams['font.family'] = 'Helvetica'

    for metric in ['volume', 'length', 'max_axial_damage_ratio']:

        if metric == 'max_axial_damage_ratio':
            df = df[df['max_axial_damage_ratio_manual_method1'] <= 1]
            df = df[df['max_axial_damage_ratio_manual_method2'] <= 1]

        # Create a figure
        fig = plt.figure(figsize=(6, 6))
        # Create a subplot
        ax = fig.add_subplot(111)
        # Plot the data (manual vs nnunet_3d) and a linear regression model fit
        # Method1 (beforeAL - phase 1) - Zurich (in dashed line)
        sns.regplot(x=metric+'_manual_method1', y=metric+'_nnunet_3d_method1', data=df[df['site'] == 'zurich'],
                    ax=ax, color='orangered', marker="^", line_kws={'ls': '--'}, scatter_kws={'s': 8})
        # Method1 (beforeAL - phase 1) - Colorado (in dashed line)
        sns.regplot(x=metric+'_manual_method1', y=metric+'_nnunet_3d_method1', data=df[df['site'] == 'colorado'],
                    ax=ax, color='deepskyblue', marker="^", line_kws={'ls': '--'}, scatter_kws={'s': 8})

        # Method2 (afterAL - phase 3) - Zurich
        sns.regplot(x=metric+'_manual_method2', y=metric+'_nnunet_3d_method2', data=df[df['site'] == 'zurich'],
                    ax=ax, color='red', scatter_kws={'s': 8})
        # Method2 (afterAL - phase 3) - Colorado
        sns.regplot(x=metric+'_manual_method2', y=metric+'_nnunet_3d_method2', data=df[df['site'] == 'colorado'],
                    ax=ax, color='darkblue', scatter_kws={'s': 8})

        # Set the title
        ax.set_title(f'{metric_to_title[metric]}', fontsize=FONT_SIZE)
        # Set the x-axis label
        ax.set_xlabel(f'Manual Ground Truth {metric_to_axis[metric]}', fontsize=FONT_SIZE)
        # Set the y-axis label
        ax.set_ylabel(f'SCIseg 3D Prediction {metric_to_axis[metric]}', fontsize=FONT_SIZE)

        if metric == 'length':
            # Set the x-axis limits
            ax.set_xlim(-5, 210)
            # Set the y-axis limits
            ax.set_ylim(-5, 210)
        elif metric == 'volume':
            # Set the x-axis limits
            ax.set_xlim(-200, 7100)
            # Set the y-axis limits
            ax.set_ylim(-200, 7100)
        else:
            # Set the x-axis limits
            ax.set_xlim(-0.1, 1.1)
            # Set the y-axis limits
            ax.set_ylim(-0.1, 1.1)

        # Draw a diagonal line
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")

        # Show grid
        ax.grid(True)

        # # Create single custom legend for whole figure with several subplots
        marker1 = plt.Line2D([0, 0], [0, 0], color='orangered', linestyle='--')
        marker2 = plt.Line2D([0, 0], [0, 0], color='red', linestyle='-')
        marker3 = plt.Line2D([0, 0], [0, 0], color='deepskyblue', linestyle='--')
        marker4 = plt.Line2D([0, 0], [0, 0], color='darkblue', linestyle='-')

        # ax.legend([marker1, marker2, marker3, marker4],
        #           ['Zurich (T2w sag) 2sites',
        #            'Zurich (T2w sag) 3sites_afterAL',
        #            'Colorado (T2w ax) 2sites',
        #            'Colorado (T2w ax) 3sites_afterAL'],
        #           numpoints=1, loc='upper left')

        ax.legend([marker1, marker2, marker3, marker4],
                  ['Site 1, Training Phase 1 (Before Active Learning)',
                   'Site 1, Training Phase 3 (After Active Learning)',
                   'Site 2, Training Phase 1 (Before Active Learning)',
                   'Site 2, Training Phase 3 (After Active Learning)'],
                  numpoints=1, loc='upper left', fontsize=FONT_SIZE-3)

        # Increase tick labels size
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE-3)

        # Save individual figures
        plt.tight_layout()

        # Save the figure
        fname_fig = os.path.join(output_dir, f'{metric}_regplot.png')
        fig.savefig(os.path.join(output_dir, fname_fig), dpi=300)
        logger.info(f'Saved {os.path.join(output_dir, fname_fig)}')
        # Close the figure
        plt.close(fig)

    # pd.set_option('display.max_colwidth', None)
    # # print subjects with max_axial_damage_ratio > 1
    # logger.info(df[(df['max_axial_damage_ratio_manual_method1'] > 1) & (df['max_axial_damage_ratio_nnunet_3d_method1'] < 1)][
    #             ['site', 'participant_id', 'max_axial_damage_ratio_manual_method1', 'max_axial_damage_ratio_nnunet_3d_method1']])
    # logger.info(df[(df['max_axial_damage_ratio_manual_method1'] > 1) & (df['max_axial_damage_ratio_nnunet_3d_method1'] < 1)][
    #             ['fname_lesion_nnunet_3d_method1']])


def compute_wilcoxon_test(df):
    """
    Compute Wilcoxon signed-rank test between nnunet_3d_method1 and nnunet_3d_method2
    :param df: dataframe with lesion metrics
    """
    # Loop across sites
    for site in ['zurich', 'colorado']:
        # Loop across metrics
        for metric in ['volume', 'length', 'max_axial_damage_ratio']:

            if metric == 'max_axial_damage_ratio':
                df = df[df['max_axial_damage_ratio_manual_method1'] <= 1]
                df = df[df['max_axial_damage_ratio_manual_method2'] <= 1]

            df_site = df[df['site'] == site]

            # Compute Wilcoxon signed-rank test between nnunet_3d_method1 and nnunet_3d_method2
            stat, pval = wilcoxon(df_site[metric + '_nnunet_3d_method1'], df_site[metric + '_nnunet_3d_method2'])
            logger.info(f'{site}: Wilcoxon test between {metric}_nnunet_3d_method1 and {metric}_nnunet_3d_method2: '
                        f'p{format_pvalue(pval)}')


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Output directory
    output_dir = os.path.join(os.getcwd(), args.o)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Created {output_dir}')

    # Dump log file there
    fname_log = f'log.txt'
    if os.path.exists(fname_log):
        os.remove(fname_log)
    fh = logging.FileHandler(os.path.join(os.path.abspath(output_dir), fname_log))
    logging.root.addHandler(fh)

    # Parse input paths
    dir_paths1 = [os.path.join(os.getcwd(), path) for path in args.method1]
    dir_paths2 = [os.path.join(os.getcwd(), path) for path in args.method2]

    # Check if the input path exists
    for dir_path in dir_paths1 + dir_paths2:
        if not os.path.exists(dir_path):
            raise ValueError(f'ERROR: {dir_path} does not exist.')

    # For each participant_id, get the lesion and spinal cord file names
    df1 = get_fnames(dir_paths1)
    df2 = get_fnames(dir_paths2)

    df_merged = pd.merge(df1, df2, on=['participant_id', 'site'], how='outer', suffixes=('_method1', '_method2'))

    # Iterate over the rows of the dataframe and read the XLS files
    for index, row in df_merged.iterrows():

        logger.info(f'Processing XLS files for {row["participant_id"]}')

        # Method 1
        # Read the XLS file with lesion metrics for lesion predicted by our 3D SCIseg nnUNet model
        df_merged = fetch_lesion_metrics(index, row, 'nnunet_3d_method1', df_merged)
        # Read the XLS file with lesion metrics for manual (GT) lesion
        df_merged = fetch_lesion_metrics(index, row, 'manual_method1', df_merged)

        # Method 2
        # Read the XLS file with lesion metrics for lesion predicted by our 3D SCIseg nnUNet model
        df_merged = fetch_lesion_metrics(index, row, 'nnunet_3d_method2', df_merged)
        # Read the XLS file with lesion metrics for manual (GT) lesion
        df_merged = fetch_lesion_metrics(index, row, 'manual_method2', df_merged)

    logger.info(f'Number of participants before the aggregation: {len(df_merged)}')

    # If a participant_id is duplicated (because the test image is presented across multiple seeds), average the
    # metrics across seeds for the same subject.
    df_merged = df_merged.groupby(['participant_id', 'site']).mean().reset_index()

    logger.info(f'Number of unique participants after the aggregation: {len(df_merged)}')

    # Replace nan with 0 for the volume_nnunet_3d column
    # nan means that there is no lesion predicted by our 3D nnUNet model;
    # therefore, metrics are assigned to 0 to plot them
    for method in '_method1', '_method2':
        df_merged['volume_nnunet_3d' + method] = df_merged['volume_nnunet_3d' + method].fillna(0)
        df_merged['length_nnunet_3d' + method] = df_merged['length_nnunet_3d' + method].fillna(0)
        df_merged['max_axial_damage_ratio_nnunet_3d' + method] = df_merged['max_axial_damage_ratio_nnunet_3d' + method].fillna(0)

    # Compute Wilcoxon signed-rank test
    compute_wilcoxon_test(df_merged)

    #  Plot data and a linear regression model fit (manual GT lesion vs lesions predicted using our 3D nnUNet model)
    generate_regplot_manual_vs_predicted(df_merged, output_dir)


if __name__ == '__main__':
    main()
