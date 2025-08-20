"""
Plot the lesion metrics obtained using different methods (manual vs automatic).

The script:
- reads CSV with lesion metrics computed using sct_analyze_lesion and aggregated across subjects
- reads XLSX file with manually measured lesion metrics (including clinical scores)
- merges the dataframes
- creates trajectory plots of clinical scores over time for each participant with baseline metrics

Example usage:
    python 04_generate_trajectory_plots.py
        -file-sct <PATH_TO_CSV_FILE>
        -file-manual <PATH_TO_XLSX_FILE>
        -o <OUTPUT_DIR>

Note: to read XLS files, you might need to install the following packages:
    pip install openpyxl

Author: Jan Valosek
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import subprocess


METRIC_TO_TITLE = {
    'midsagittal_length': 'Midsagittal Lesion Length [mm]',
    'midsagittal_width': 'Midsagittal Lesion Width [mm]',
    'ventral_tissue_bridge': 'Midsagittal Ventral Tissue Bridges [mm]',
    'dorsal_tissue_bridge': 'Midsagittal Dorsal Tissue Bridges [mm]',
    'total_tissue_bridge': 'Midsagittal Total Tissue Bridges [mm]',
    'dorsal_bridge_ratio': 'Midsagittal Dorsal Tissue Bridge Ratio [%]',
    'ventral_bridge_ratio': 'Midsagittal Ventral Tissue Bridge Ratio [%]',
}

METHOD_TO_TITLE = {
    'GT': 'Semi-automatic (manual lesion masks + SCT)',
    'SCIsegV2': 'Automatic (SCIsegV2 + SCT)'
}

METHOD_TO_FNAME = {
    'GT': 'semiautomatic',
    'SCIsegV2': 'automatic'
}

CLINICAL_SCORES_TO_AXES = {
    'uems': 'UEMS',
    'lems': 'LEMS',
    'ms': 'Total Motor Score',
    'pp': 'Pinprick Score',
    'lt': 'Light-Touch Score'
}

CLINICAL_SCORES_MAX = {
    'uems': 50,  # Maximum UEMS score
    'lems': 50,  # Maximum LEMS score
    'ms': 100,    # Maximum Total Motor Score
    'pp': 112,   # Maximum Pinprick Score
    'lt': 112    # Maximum Light-Touch Score
}

FONT_SIZE = 12


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='Read CSV files with lesion metrics computed using sct_analyze_lesion and XLSX file with manually'
                    'measured metrics and create figures.',
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
        help='Absolute path to an XLSX file with manually measured lesion metrics and baseline clinical scores. '
    )
    parser.add_argument(
        '-o',
        required=True,
        type=str,
        help='Path to the output folder where figures will be saved. '
    )

    return parser


def read_file_manual(file):
    """
    Read the XLSX file with manually measured metrics and clinical scores.
    :param file: str: path to the XLSX file
    :return df_manual: pandas DataFrame: dataframe with manually measured lesion metrics and clinical scores
    """
    df_manual = pd.read_excel(file)
    # Drop rows where 'participant_id' is NaN or 'exclude' -- rows with comments
    df_manual = df_manual.dropna(subset=['participant_id'])
    df_manual = df_manual[df_manual['participant_id'] != 'exclude']
    # If session_id is nan in the manual file, set it to 'ses-01'
    df_manual['session_id'] = df_manual['session_id'].fillna('ses-01')

    # Sum up ventral and dorsal tissue bridges to get total tissue bridge
    df_manual['total_tissue_bridge'] = df_manual['ventral_tissue_bridge'] + df_manual['dorsal_tissue_bridge']
    # Rename columns to distinguish manual metrics from SCT metrics
    df_manual.rename(columns={'midsagittal_length': 'midsagittal_length_manual',
                              'midsagittal_width': 'midsagittal_width_manual',
                              'ventral_tissue_bridge': 'ventral_tissue_bridge_manual',
                              'dorsal_tissue_bridge': 'dorsal_tissue_bridge_manual',
                              'total_tissue_bridge': 'total_tissue_bridge_manual'},
                     inplace=True)

    # Compute tissue bridge ratios
    df_manual['dorsal_bridge_ratio_manual'] = df_manual.apply(
        lambda row: (row['dorsal_tissue_bridge_manual'] / row['total_tissue_bridge_manual'] * 100)
        if row['total_tissue_bridge_manual'] > 0 else 0, axis=1)
    df_manual['ventral_bridge_ratio_manual'] = df_manual.apply(
        lambda row: (row['ventral_tissue_bridge_manual'] / row['total_tissue_bridge_manual'] * 100)
        if row['total_tissue_bridge_manual'] > 0 else 0, axis=1)

    # Remove any strings from the 'mri_time_since_injury' column
    if 'mri_time_since_injury' in df_manual.columns:
        df_manual['mri_time_since_injury'] = df_manual['mri_time_since_injury'].astype(str).str.extract(r'(\d+)').astype(int)

    # Convert any string clinical score columns to numeric
    for col in df_manual.columns:
        if col.startswith(tuple(CLINICAL_SCORES_TO_AXES.keys())):
            df_manual[col] = pd.to_numeric(df_manual[col], errors='coerce')

    # Drop 'comment' column and unnamed columns
    df_manual = df_manual.drop(columns=['comment'], errors='ignore')
    unnamed_cols = [col for col in df_manual.columns if 'Unnamed' in col]
    df_manual = df_manual.drop(columns=unnamed_cols, errors='ignore')

    # Reorder the columns to have participant_id, session_id, and mri_time_since_injury first, followed by manual
    # metrics (_manual suffix)
    cols = ['participant_id', 'session_id', 'mri_time_since_injury'] + \
           [col for col in df_manual.columns if col.endswith('_manual')] + \
           [col for col in df_manual.columns if not col.endswith('_manual') and col not in ['participant_id', 'session_id', 'mri_time_since_injury']]
    df_manual = df_manual[cols]

    print(f'Read {len(df_manual)} rows from the manual metrics file: {file}')
    return df_manual


def normalize_sensorimotor_scores(df):
    """
    Normalize clinical scores (uems, lems, ms, pp, lt) between follow-ups.
    Works with subjects who have either baseline or 1m as their first exam.
    For each subject:
    1. Identifies the first available time point (baseline or 1m)
    2. Computes maximal improvable score from this first time point
    3. Normalizes subsequent scores by dividing improvement by maximal improvable score

    :param df: pandas DataFrame with baseline lesion metrics and clinical scores across
    multiple time points
    :return df: pandas DataFrame with normalized clinical scores
    """
    # Get unique participant IDs
    participants = df['participant_id'].unique()

    # Time points in order
    time_points = ['bl', '1m', '3m', '6m', '12m']

    # Loop through each participant
    for participant in participants:
        # Get data for the current participant
        participant_data = df[df['participant_id'] == participant]

        # Process each clinical score
        for score in CLINICAL_SCORES_TO_AXES.keys():
            # Find first available time point for this score
            first_tp = None
            first_value = None

            # Check time points in order (baseline first, then 1m)
            for tp in time_points:
                col_name = f"{score}_{tp}"
                if col_name in participant_data.columns and not pd.isna(participant_data[col_name].values[0]):
                    first_tp = tp
                    first_value = participant_data[col_name].values[0]
                    break

            # Skip if no data available for this score
            if first_tp is None:
                continue

            # Get maximum possible score for this clinical measure
            max_score = CLINICAL_SCORES_MAX[score]

            # Calculate maximal improvable score (difference between max possible and first value)
            max_improvable = max_score - first_value

            # Find all subsequent time points to normalize
            subsequent_tps = time_points[time_points.index(first_tp) + 1:]

            # Normalize each follow-up time point after the first available one
            for tp in subsequent_tps:
                follow_up_col = f"{score}_{tp}"
                if follow_up_col in participant_data.columns and not pd.isna(participant_data[follow_up_col].values[0]):
                    # Calculate improvement from first time point
                    follow_up_value = participant_data[follow_up_col].values[0]
                    improvement = follow_up_value - first_value

                    # Create new column for normalized score
                    normalized_col = f"{follow_up_col}_improvement_normalized"

                    # Normalize improvement by maximal improvable score
                    if max_improvable <= 0:  # If no room for improvement, set to 0
                        df.loc[participant_data.index, normalized_col] = 0  # or should I use `np.nan`?
                    else:
                        normalized_improvement = improvement / max_improvable
                        df.loc[participant_data.index, normalized_col] = normalized_improvement

    return df


def read_file_sct(file_sct):
    df_sct = pd.read_csv(file_sct)
    # Rename columns to match the manual metrics
    df_sct.rename(columns={'length_interpolated_midsagittal_slice': 'midsagittal_length',
                           'width_interpolated_midsagittal_slice': 'midsagittal_width',
                           'interpolated_dorsal_bridge_width': 'dorsal_tissue_bridge',
                           'interpolated_ventral_bridge_width': 'ventral_tissue_bridge',
                           'interpolated_total_bridge_width': 'total_tissue_bridge'},
                  inplace=True)
    # Add suffix to all columns except participant_id and session_id
    df_sct = df_sct.add_suffix('_sct')
    df_sct.rename(columns={'participant_id_sct': 'participant_id', 'session_id_sct': 'session_id'}, inplace=True)

    print(f'Read {len(df_sct)} rows from the SCT metrics file: {file_sct}')
    return df_sct


def format_pvalue(p_value, alpha=0.05):
    """
    Format p-value for display.
    """
    if p_value < 0.001:
        return 'p < 0.001'
    elif p_value < alpha:
        return f'p < {alpha}'
    else:
        return f'p = {p_value:.3f}'


def combine_plot(figure_type, num_subjects, output_dir):
    """
    Combine all the plots into a single figure using bash convert command
    This requires ImageMagick to be installed
    :param figure_type: str: type of the figure to combine (e.g., 'scatterplot', or 'diffplot')
    :param num_subjects: int: number of subjects in the dataframe
    :param output_dir: str: output directory where the combined figure will be saved
    """

    # Create 'combined' directory if it does not exist
    combined_dir = os.path.join(output_dir, 'combined')
    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)

    print(f"Combining {figure_type}s into a single figure...")
    # 1 row, 3 columns:
    cmd_combine = f"convert {os.path.join(output_dir, f'{figure_type}_midsagittal_length_{num_subjects}subjects.png')} " \
          f"{os.path.join(output_dir, f'{figure_type}_midsagittal_width_{num_subjects}subjects.png')} " \
          f"{os.path.join(output_dir, f'{figure_type}_total_tissue_bridge_{num_subjects}subjects.png')} " \
          f"+append {os.path.join(combined_dir, f'{figure_type}_combined_{num_subjects}subjects.png')}"
    # 3 rows, 3 columns:
    # # First row: midsagittal_length and midsagittal_width
    # cmd_row1 = f"convert {os.path.join(output_dir, f'{figure_type}_midsagittal_length_{num_subjects}subjects.png')} " \
    #            f"{os.path.join(output_dir, f'{figure_type}_midsagittal_width_{num_subjects}subjects.png')} " \
    #            f"+append {os.path.join(output_dir, f'temp1_{num_subjects}subjects.png')}"
    # # Second row: ventral_tissue_bridge, dorsal_tissue_bridge, and total_tissue_bridge
    # cmd_row2 = f"convert {os.path.join(output_dir, f'{figure_type}_ventral_tissue_bridge_{num_subjects}subjects.png')} " \
    #            f"{os.path.join(output_dir, f'{figure_type}_dorsal_tissue_bridge_{num_subjects}subjects.png')} " \
    #            f"{os.path.join(output_dir, f'{figure_type}_total_tissue_bridge_{num_subjects}subjects.png')} " \
    #            f"+append {os.path.join(output_dir, f'temp2_{num_subjects}subjects.png')}"
    # # Third row: ventral_bridge_ratio and dorsal_bridge_ratio
    # cmd_row3 = f"convert {os.path.join(output_dir, f'{figure_type}_ventral_bridge_ratio_{num_subjects}subjects.png')} " \
    #            f"{os.path.join(output_dir, f'{figure_type}_dorsal_bridge_ratio_{num_subjects}subjects.png')} " \
    #            f"+append {os.path.join(output_dir, f'temp3_{num_subjects}subjects.png')}"
    # # Combine all rows
    # cmd_combine = f"convert {os.path.join(output_dir, f'temp1_{num_subjects}subjects.png')} " \
    #               f"{os.path.join(output_dir, f'temp2_{num_subjects}subjects.png')} " \
    #               f"{os.path.join(output_dir, f'temp3_{num_subjects}subjects.png')} " \
    #               f"-append {os.path.join(combined_dir, f'{figure_type}_combined_{num_subjects}subjects.png')}; " \
    #               f"rm {os.path.join(output_dir, f'temp1_{num_subjects}subjects.png')} " \
    #               f"{os.path.join(output_dir, f'temp2_{num_subjects}subjects.png')} " \
    #               f"{os.path.join(output_dir, f'temp3_{num_subjects}subjects.png')}"
    # # Execute the commands
    # subprocess.run(cmd_row1, shell=True)
    # subprocess.run(cmd_row2, shell=True)
    # subprocess.run(cmd_row3, shell=True)
    subprocess.run(cmd_combine, shell=True)
    print(f"Combined {figure_type} saved as {os.path.join(combined_dir, f'{figure_type}_combined_{num_subjects}subjects.png')}")


def create_trajectory_plots(df, output_dir, method):
    """
    Create an individual trajectory plot showing clinical scores across time points for each participant,
    with lines colored by baseline lesion metrics.
    Also creates plots showing normalized improvement scores for follow-up time points.

    :param df: pandas dataframe with baseline lesion metrics and clinical scores across
    multiple time points
    :param output_dir: output directory
    :method: str: method ('GT' or 'SCIsegV2')
    """
    # Set font to Arial
    plt.rcParams['font.sans-serif'] = 'Arial'

    # Define metric thresholds for stratification (using equal increments)
    metric_thresholds = {
        'midsagittal_length': [0, 10, 20],
        'midsagittal_width': [0, 3, 6],
        'ventral_tissue_bridge': [0, 1],
        'dorsal_tissue_bridge': [0, 1],
        'total_tissue_bridge': [0, 1, 2],
        'dorsal_bridge_ratio': [0, 50],
        'ventral_bridge_ratio': [0, 50]
    }

    # Define colors for each group
    group_colors = ['blue', 'green', 'red']

    # Define time points with their actual time values in months from baseline
    time_point_mapping = {
        'bl': {'order': 0, 'months': 0},     # baseline = 0 months
        '1m': {'order': 1, 'months': 1},     # 1 month
        '3m': {'order': 2, 'months': 3},     # 3 months
        '6m': {'order': 3, 'months': 6},     # 6 months
        '12m': {'order': 4, 'months': 12}    # 12 months
    }
    # Sort time points in a logical order
    time_points = sorted(list(time_point_mapping.keys()), key=lambda x: time_point_mapping.get(x, {}).get('order', 99))

    # PART 1: Raw trajectory plots
    create_raw_trajectory_plots(df, group_colors, method, metric_thresholds, output_dir,
                                time_point_mapping,  time_points)

    # PART 2: Normalized improvement scores trajectory plots
    create_normalized_trajectory_plots(df, group_colors, method, metric_thresholds, output_dir,
                                       time_point_mapping, time_points)


def create_raw_trajectory_plots(df, group_colors, method, metric_thresholds, output_dir, time_point_mapping,
                                time_points):
    """
    Create trajectory plots for raw (i.e., non-normalized) clinical scores across time points for each participant,
    with lines colored by baseline lesion metrics.

    :param df: pandas dataframe with baseline lesion metrics and clinical scores across
    multiple time points
    :param group_colors: list of colors for each group
    :param method: str: method ('GT' or 'SCIsegV2')
    :param metric_thresholds: dict: thresholds for each metric to create groups
    :param output_dir: output directory
    :param time_point_mapping: dict: mapping of time points to their actual time values in months from baseline
    :param time_points: list of time points in logical order
    """

    # Get the actual month values for x-axis positioning
    time_points_months = [time_point_mapping[tp]['months'] for tp in time_points]

    # Loop over each clinical score
    for score in CLINICAL_SCORES_TO_AXES.keys():
        # Filtering for UEMS
        if score == 'uems':
            # For UEMS, keep only subjects with 'tetrapara_bl' == 0
            #   0: tetraplegic
            #   1: paraplegic -- max UEMS at baseline (no impairment)
            df_plot = df[df['tetrapara_bl'] == 0]
        # Keeping all subjects for LEMS and other scores
        else:
            df_plot = df

        # Loop over each lesion metric
        for metric in METRIC_TO_TITLE.keys():
            # Create a figure for all participants
            fig, ax = plt.subplots(figsize=(10, 6))

            # Use lesion metrics computed by SCT (and not manually measured ones)
            # Note: there are still two possible options controlled by the `method` variable:
            #   1. 'GT' - manual lesion masks + sct_analyze_lesion
            #   2. 'SCIsegV2' - SCIsegV2 lesion masks + sct_analyze_lesion
            metric_name = f'{metric}_sct'

            # Get the thresholds for this metric
            thresholds = metric_thresholds[metric]

            # Create groups for each threshold range
            group_ids = [[] for _ in range(len(thresholds))]
            # Collect data for mean trajectories per group
            group_data = [{tp: [] for tp in time_points} for _ in range(len(thresholds))]

            # Process each participant
            for participant_id in df['participant_id'].unique():
                # Get participant data
                participant_data = df[df['participant_id'] == participant_id]
                # Check if participant has clinical data and the metric
                if (participant_id not in df_plot['participant_id'].values or
                        metric_name not in participant_data.columns or
                        pd.isna(participant_data[metric_name].values[0])):
                    continue

                participant_clinical = df_plot[df_plot['participant_id'] == participant_id]

                # Get metric value for this participant (for coloring)
                metric_value = participant_data[metric_name].values[0]

                # Determine which group this participant belongs to
                group_idx = 0
                for i in range(len(thresholds) - 1):
                    if thresholds[i] <= metric_value < thresholds[i + 1]:
                        group_idx = i
                        break
                if metric_value >= thresholds[-1]:
                    group_idx = len(thresholds) - 1

                # Add participant to the group
                group_ids[group_idx].append(participant_id)

                # Get clinical score values across time points
                time_points_present = [tp for tp in time_points if f'{score}_{tp}' in participant_clinical.columns]

                if not time_points_present:
                    continue

                # Get values for this score across time points
                time_values = []
                score_values = []

                for tp in time_points_present:
                    col_name = f'{score}_{tp}'
                    if col_name in participant_clinical.columns and not pd.isna(
                            participant_clinical[col_name].values[0]):
                        val = float(participant_clinical[col_name].values[0])  # Ensure value is float
                        # Use actual month values for x-axis
                        time_values.append(time_point_mapping[tp]['months'])
                        score_values.append(val)

                        # Add to group data for mean trajectory
                        group_data[group_idx][tp].append(val)

                if len(time_values) < 2:  # Need at least 2 points to draw a line
                    continue

                # # Trajectory lines - match the group color
                # ax.plot(time_values, score_values, 'o-', alpha=0.3,
                #         color=group_colors[group_idx],
                #         linewidth=0.5, markersize=0)

            # Calculate and plot mean ± confidence interval (CI) trajectories for each group
            for tp in time_points:
                # Get x position in months
                tp_month = time_point_mapping[tp]['months']

                for group_idx in range(len(thresholds)):
                    # Get group values for this time point
                    group_values = group_data[group_idx][tp]

                    if group_values:
                        # Ensure all values are numeric
                        group_values = [float(val) for val in group_values]
                        group_mean = np.mean(group_values)
                        # Calculate 95% confidence interval
                        group_ci = 1.96 * np.std(group_values) / np.sqrt(len(group_values)) if len(
                            group_values) > 1 else 0

                        # Create group label based on the threshold range
                        unit = '%' if 'ratio' in metric else 'mm'
                        if group_idx == 0:
                            # First group
                            label = f'<{thresholds[1]} {unit} (n={len(group_ids[group_idx])})'
                        elif group_idx == len(thresholds) - 1:
                            # Intermediate groups
                            label = f'≥{thresholds[group_idx]} {unit} (n={len(group_ids[group_idx])})'
                        else:
                            # Last group
                            label = f'{thresholds[group_idx]}-{thresholds[group_idx + 1]} {unit} (n={len(group_ids[group_idx])})'

                        # Only show label in legend for the first time point (to avoid duplicates)
                        ax.errorbar(tp_month, group_mean, yerr=group_ci,
                                    fmt='o', color=group_colors[group_idx], ecolor=group_colors[group_idx],
                                    markersize=5, capsize=5,
                                    label=label if tp == '1m' else "")

            # Connect mean points with lines for each group
            for group_idx in range(len(thresholds)):
                mean_x = []
                mean_y = []

                for tp in time_points:
                    tp_month = time_point_mapping[tp]['months']
                    group_values = group_data[group_idx][tp]

                    if group_values:
                        # Ensure all values are numeric
                        group_values = [float(val) for val in group_values]
                        mean_x.append(tp_month)
                        mean_y.append(np.mean(group_values))

                if len(mean_x) > 1:
                    ax.plot(mean_x, mean_y, '-', color=group_colors[group_idx], linewidth=2.5)

            # Set labels and title
            ax.set_title(f'{CLINICAL_SCORES_TO_AXES[score]} over time stratified by '
                         f'{METRIC_TO_TITLE[metric].split("[")[0]}', fontsize=FONT_SIZE + 2)
            ax.set_xlabel('Time Point', fontsize=FONT_SIZE)
            ax.set_ylabel(f'{CLINICAL_SCORES_TO_AXES[score]}', fontsize=FONT_SIZE)

            # Set x-ticks at the actual month points (0, 1, 3, 6, 12)
            ax.set_xticks(time_points_months)
            ax.set_xticklabels(['Baseline', 'M1', 'M3', 'M6', 'M12'], fontsize=FONT_SIZE)

            # Add minor ticks only at the labeled tick positions for better visualization
            ax.tick_params(axis='x', which='minor', bottom=True, length=4)
            # Place minor ticks at the same positions as the major ticks
            ax.set_xticks(time_points_months, minor=True)

            # Add legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=FONT_SIZE - 2, framealpha=0.9,
                      title=f'{METRIC_TO_TITLE[metric].split("[")[0]}\nmean ± CI', title_fontsize=FONT_SIZE - 2)

            # Remove the top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Save the plot
            plt.tight_layout()
            num_subjects = len(df_plot)
            figure_fname = os.path.join(output_dir,
                                        f'{METHOD_TO_FNAME[method]}_trajectory_plot_{score}_{metric}_{num_subjects}subjects.png')
            plt.savefig(figure_fname, dpi=300)
            print(f'Trajectory plot for {score} by {metric} saved as {figure_fname}')
            plt.close()
        # Combine individual trajectory plots for this score
        combine_plot(f'{METHOD_TO_FNAME[method]}_trajectory_plot_{score}', num_subjects, output_dir)


def create_normalized_trajectory_plots(df, group_colors, method, metric_thresholds, output_dir,
                                       time_point_mapping, time_points):
    """
    Create trajectory plots for normalized clinical scores across follow-up time points for each participant,
    with lines colored by baseline lesion metrics.
    This function assumes that the normalized improvement scores have already been computed and added to the dataframe
    using the `normalize_sensorimotor_scores` function.

    :param df: pandas dataframe with baseline lesion metrics and clinical scores across multiple time points
    :param group_colors: list of colors for each group
    :param method: str: method ('GT' or 'SCIsegV2')
    :param metric_thresholds: dict: thresholds for each metric to create groups
    :param output_dir: output directory
    :param time_point_mapping: dict: mapping of time points to their actual time values in months from baseline
    :param time_points: list of time points in logical order
    """
    # Loop over each clinical score
    for score in CLINICAL_SCORES_TO_AXES.keys():
        # Filtering for UEMS
        if score == 'uems':
            # For UEMS, keep only subjects with 'tetrapara_bl' == 0
            #   0: tetraplegic
            #   1: paraplegic -- max UEMS at baseline (no impairment)
            df_plot = df[df['tetrapara_bl'] == 0]
        # Keeping all subjects for LEMS and other scores
        else:
            df_plot = df

        # Get the follow-up time points (excluding baseline)
        follow_up_time_points = [tp for tp in time_points if tp != 'bl']

        # Loop over each lesion metric
        for metric in METRIC_TO_TITLE.keys():
            # Create a figure for all participants
            fig, ax = plt.subplots(figsize=(10, 6))

            # Use lesion metrics computed by SCT (and not manually measured ones)
            # Note: there are still two possible options controlled by the `method` variable:
            #   1. 'GT' - manual lesion masks + sct_analyze_lesion
            #   2. 'SCIsegV2' - SCIsegV2 lesion masks + sct_analyze_lesion
            metric_name = f'{metric}_sct'

            # Get the thresholds for this metric
            thresholds = metric_thresholds[metric]

            # Create groups for each threshold range
            group_ids = [[] for _ in range(len(thresholds))]

            # Collect data for mean normalized improvements per group and follow-up
            group_normalized_data = [{tp: [] for tp in follow_up_time_points} for _ in range(len(thresholds))]

            # Process each participant
            for participant_id in df['participant_id'].unique():
                # Get participant data
                participant_data = df[df['participant_id'] == participant_id]
                # Check if participant has clinical data and the metric
                if (participant_id not in df_plot['participant_id'].values or
                        metric_name not in participant_data.columns or
                        pd.isna(participant_data[metric_name].values[0])):
                    continue

                participant_clinical = df_plot[df_plot['participant_id'] == participant_id]

                # Get metric value for this participant (for grouping)
                metric_value = participant_data[metric_name].values[0]

                # Determine which group this participant belongs to
                group_idx = 0
                for i in range(len(thresholds) - 1):
                    if thresholds[i] <= metric_value < thresholds[i + 1]:
                        group_idx = i
                        break
                if metric_value >= thresholds[-1]:
                    group_idx = len(thresholds) - 1

                # Add participant to the group
                group_ids[group_idx].append(participant_id)

                # Get normalized improvement values for each follow-up time point
                time_values = []
                normalized_improvement_values = []

                for tp in follow_up_time_points:
                    normalized_col = f"{score}_{tp}_improvement_normalized"

                    if normalized_col in participant_clinical.columns and not pd.isna(
                            participant_clinical[normalized_col].values[0]):
                        val = float(participant_clinical[normalized_col].values[0])
                        # Use actual month values for x-axis
                        time_values.append(time_point_mapping[tp]['months'])
                        normalized_improvement_values.append(val)

                        # Add to group data for mean trajectory
                        group_normalized_data[group_idx][tp].append(val)

                if len(time_values) < 1:  # Need at least 1 point
                    continue

                # # Trajectory lines - only if there are multiple points
                # if len(time_values) > 1:
                #     ax.plot(time_values, normalized_improvement_values, 'o-', alpha=0.3,
                #             color=group_colors[group_idx],
                #             linewidth=0.5, markersize=0)

            # Calculate and plot mean ± confidence interval (CI) for normalized improvements
            for tp in follow_up_time_points:
                # Get x position in months
                tp_month = time_point_mapping[tp]['months']

                for group_idx in range(len(thresholds)):
                    # Get group values for this time point
                    group_values = group_normalized_data[group_idx][tp]

                    if group_values:
                        # Ensure all values are numeric
                        group_values = [float(val) for val in group_values]
                        group_mean = np.mean(group_values)
                        # Calculate 95% confidence interval
                        group_ci = 1.96 * np.std(group_values) / np.sqrt(len(group_values)) if len(
                            group_values) > 1 else 0

                        # Create group label based on the threshold range
                        unit = '%' if 'ratio' in metric else 'mm'
                        if group_idx == 0:
                            # First group
                            label = f'<{thresholds[1]} {unit} (n={len(group_ids[group_idx])})'
                        elif group_idx == len(thresholds) - 1:
                            # Intermediate groups
                            label = f'≥{thresholds[group_idx]} {unit} (n={len(group_ids[group_idx])})'
                        else:
                            # Last group
                            label = f'{thresholds[group_idx]}-{thresholds[group_idx + 1]} {unit} (n={len(group_ids[group_idx])})'

                        # Only show label in legend for the first time point (to avoid duplicates)
                        ax.errorbar(tp_month, group_mean, yerr=group_ci,
                                    fmt='o', color=group_colors[group_idx], ecolor=group_colors[group_idx],
                                    markersize=5, capsize=5,
                                    label=label if tp == '1m' else "")

            # Connect mean points with lines for each group
            for group_idx in range(len(thresholds)):
                mean_x = []
                mean_y = []

                for tp in follow_up_time_points:
                    tp_month = time_point_mapping[tp]['months']
                    group_values = group_normalized_data[group_idx][tp]

                    if group_values:
                        # Ensure all values are numeric
                        group_values = [float(val) for val in group_values]
                        mean_x.append(tp_month)
                        mean_y.append(np.mean(group_values))

                if len(mean_x) > 1:
                    ax.plot(mean_x, mean_y, '-', color=group_colors[group_idx], linewidth=2.5)

            # Set labels and title
            ax.set_title(f'Normalized improvement in {CLINICAL_SCORES_TO_AXES[score]} stratified by '
                         f'{METRIC_TO_TITLE[metric].split("[")[0]}', fontsize=FONT_SIZE + 2)
            ax.set_xlabel('Time Point', fontsize=FONT_SIZE)
            ax.set_ylabel(f'Normalized Improvement', fontsize=FONT_SIZE)

            # Add a horizontal line at y=0 (no improvement)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

            # Tweak y-axis limits for sensory scores
            if score in ['pp', 'lt']:
                ax.set_ylim(-0.8, 1.2)
            else:
                ax.set_ylim(-0.2, 1.2)

            # Set x-ticks at follow-up month points (1, 3, 6, 12)
            follow_up_months = [time_point_mapping[tp]['months'] for tp in follow_up_time_points]
            ax.set_xticks(follow_up_months)
            ax.set_xticklabels(['M1', 'M3', 'M6', 'M12'], fontsize=FONT_SIZE)

            # Add minor ticks only at the labeled tick positions for better visualization
            ax.tick_params(axis='x', which='minor', bottom=True, length=4)
            # Place minor ticks at the same positions as the major ticks
            ax.set_xticks(follow_up_months, minor=True)

            # Add legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=FONT_SIZE - 2, framealpha=0.9,
                      title=f'{METRIC_TO_TITLE[metric].split("[")[0]}\nmean ± CI', title_fontsize=FONT_SIZE - 2)

            # Remove the top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Save the plot
            plt.tight_layout()
            num_subjects = len(df_plot)
            figure_fname = os.path.join(output_dir,
                                        f'{METHOD_TO_FNAME[method]}_normalized_improvement_{score}_{metric}_{num_subjects}subjects.png')
            plt.savefig(figure_fname, dpi=300)
            print(f'Normalized improvement plot for {score} by {metric} saved as {figure_fname}')
            plt.close()
        # Combine individual trajectory plots for this score
        combine_plot(f'{METHOD_TO_FNAME[method]}_normalized_improvement_{score}', num_subjects, output_dir)


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
    # Get 'GT' or 'SCIsegV2' method from the filename
    method = file_sct.split('_')[-1].replace('.csv','')

    #----------------
    # XLSX file with manually measured lesion metrics and clinical scores
    #----------------
    df_manual = read_file_manual(file_manual)

    #----------------
    # Merge the dataframes
    #----------------
    df = pd.merge(df_sct, df_manual, on=['participant_id', 'session_id'])

    #----------------
    # Normalize sensorimotor scores
    #----------------
    print(f'Number of subjects: {df.shape[0]}')
    df = normalize_sensorimotor_scores(df)

    #----------------
    # Filter subjects based on MRI time since injury
    #----------------
    # Convert mri_time_since_injury to numeric (in days)
    df['mri_time_since_injury'] = pd.to_numeric(df['mri_time_since_injury'])
    print(f'Number of subjects before filtering by MRI time since injury: {df.shape[0]}')
    # Keep only subjects with mri_time_since_injury (in days) from 12 days to 2 months
    df = df[(df['mri_time_since_injury'] >= 12) & (df['mri_time_since_injury'] <= 133)]
    print(f'Number of subjects after filtering by MRI time since injury: {df.shape[0]}')

    # Drop rows with NaN values in the lesion metrics
    df = df.dropna(subset=[f'{metric}_sct' for metric in METRIC_TO_TITLE.keys()])
    print(f'Number of subjects after dropping NaN values: {df.shape[0]}')

    #----------------
    # Plotting
    #----------------
    output_dir = args.o
    os.makedirs(output_dir, exist_ok=True)

    #----------------
    # Clinical scores and baseline metrics over time
    #----------------
    create_trajectory_plots(df, output_dir, method)


if __name__ == '__main__':
    main()
