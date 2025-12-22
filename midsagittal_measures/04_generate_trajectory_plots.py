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

from matplotlib.lines import Line2D

from utils import read_csv_file_with_lesion_metrics

METRIC_TO_TITLE = {
    'midsagittal_length': 'Lesion Length',
    'midsagittal_width': 'Lesion Width',
    # 'ventral_tissue_bridge': 'Midsagittal Ventral Tissue Bridges [mm]',
    # 'dorsal_tissue_bridge': 'Midsagittal Dorsal Tissue Bridges [mm]',
    'total_tissue_bridge': 'Total Tissue Bridges',
    # 'dorsal_bridge_ratio': 'Midsagittal Dorsal Tissue Bridge Ratio [%]',
    # 'ventral_bridge_ratio': 'Midsagittal Ventral Tissue Bridge Ratio [%]',
}


METHOD_TO_FNAME = {
    'GT': 'semiautomatic',
    'SCIsegV2': 'automatic'
}

CLINICAL_SCORES_TO_AXES = {
    'uems': 'UEMS',
    'lems': 'Lower Extremity Motor Score',
    'ms': 'Total Motor Score',
    'pp': 'Pinprick Score',
    'lt': 'Light-Touch Score'
}


FONT_SIZE = 16


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
        '-i',
        required=True,
        type=str,
        help='Absolute path to a CSV file with lesion metrics with _sct and _manual suffixes.'
    )
    parser.add_argument(
        '-file-clinical-nisci',
        required=True,
        type=str,
        help='Absolute path to a XLSX file with participant clinical data for NISCI (clinical_scores.xlsx).'
    )
    parser.add_argument(
        '-file-clinical-sci-zurich',
        required=True,
        type=str,
        help='Absolute path to a XLSX file with participant clinical data for sci-zurich.'
    )
    parser.add_argument(
        '-o',
        required=True,
        type=str,
        help='Path to the output folder where figures and tables will be saved.'
    )

    return parser



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


def create_trajectory_plots(df, output_dir):
    """
    Create an individual trajectory plot showing clinical scores across time points for each participant,
    with lines colored by baseline lesion metrics.
    :param df: pandas dataframe with baseline lesion metrics and clinical scores across multiple time points
    :param output_dir: output directory
    """
    # Set font to Arial
    plt.rcParams['font.sans-serif'] = 'Arial'

    # Use thresholds identified by URP-CTREE
    metric_thresholds = {
        'lems': {
            'midsagittal_width': [5.878],
            'total_tissue_bridge': [0.17]
        },
        'ms': {
            'midsagittal_width': [5.878],
            'total_tissue_bridge': [0.17]
        },
        'pp': {
            'midsagittal_width': [6.413]
        },
        'lt': {
            'midsagittal_width': [6.413]
        }
    }

    group_colors = ['blue', 'green', 'red']

    # Define time points with their actual time values in months from baseline
    time_point_mapping = {
        'bl': {'order': 0, 'months': 0},     # baseline = 0 months
        '1m': {'order': 1, 'months': 1},     # 1 month
        '3m': {'order': 2, 'months': 3},     # 3 months
        '6m': {'order': 3, 'months': 6},     # 6 months
    }
    # Sort time points in a logical order
    time_points = sorted(list(time_point_mapping.keys()), key=lambda x: time_point_mapping.get(x, {}).get('order', 99))

    # PART 1: Raw trajectory plots
    create_raw_trajectory_plots(df, group_colors, metric_thresholds, output_dir, time_point_mapping, time_points)


def create_raw_trajectory_plots(df, group_colors, metric_thresholds, output_dir, time_point_mapping, time_points):
    """
    Create trajectory plots for raw (i.e., non-normalized) clinical scores across time points for each participant,
    with lines colored by baseline lesion metrics.

    :param df: pandas dataframe with baseline lesion metrics and clinical scores across multiple time points
    :param group_colors: list of colors for each group
    :param metric_thresholds: dict: thresholds for each metric to create groups
    :param output_dir: output directory
    :param time_point_mapping: dict: mapping of time points to their actual time values in months from baseline
    :param time_points: list of time points in logical order
    """

    # Loop over clinical scores (e.g., 'lems', 'ms', ...)
    for score, metrics in metric_thresholds.items():
        # Create groups based on thresholds
        # If there's only one metric, create 2 groups
        # If there are multiple metrics, create 3 groups using hierarchical splitting
        metric_names = list(metrics.keys())

        # Single metric - 2 groups
        if len(metric_names) == 1:
            metric = metric_names[0]
            metric_name = f'{metric}_sct'
            thresholds = metric_thresholds[score][metric]
            threshold = thresholds[0]

            # Create a figure for all participants
            fig, ax = plt.subplots(figsize=(10, 6))

            num_groups = 2
            group_ids = [[] for _ in range(num_groups)]
            group_data = [{tp: [] for tp in time_points} for _ in range(num_groups)]

            # Process each participant
            for participant_id in df['participant_id'].unique():
                # Get participant data
                participant_data = df[df['participant_id'] == participant_id]
                # Check if participant has clinical data and the metric column exists
                if (participant_id not in df['participant_id'].values or
                        metric_name not in participant_data.columns):
                    continue

                participant_clinical = df[df['participant_id'] == participant_id]

                # Get metric value for this participant (for coloring)
                metric_value = participant_data[metric_name].values[0]

                # Determine which group this participant belongs to
                # If metric value is NaN, assign to group 0
                if pd.isna(metric_value):
                    group_idx = 0
                else:
                    group_idx = 0 if metric_value <= threshold else 1

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

            # Generate labels and plot for single metric case
            plot_trajectory_groups(ax, group_data, time_points, time_point_mapping,
                                 score, metric, group_colors, num_groups,
                                 [(threshold, None)], single_metric=True)

        # Multiple metrics - 3 groups using hierarchical splitting
        else:
            metric1 = metric_names[0]  # First metric for initial split
            metric2 = metric_names[1]  # Second metric for subdivision
            threshold1 = metric_thresholds[score][metric1][0]
            threshold2 = metric_thresholds[score][metric2][0]

            metric1_name = f'{metric1}_sct'
            metric2_name = f'{metric2}_sct'

            # Create a figure for all participants
            fig, ax = plt.subplots(figsize=(10, 6))

            num_groups = 3
            group_ids = [[] for _ in range(num_groups)]
            group_data = [{tp: [] for tp in time_points} for _ in range(num_groups)]

            # Process each participant
            for participant_id in df['participant_id'].unique():
                # Get participant data
                participant_data = df[df['participant_id'] == participant_id]
                # Check if participant has clinical data and both metric columns exist
                if (participant_id not in df['participant_id'].values or
                        metric1_name not in participant_data.columns or
                        metric2_name not in participant_data.columns):
                    continue

                participant_clinical = df[df['participant_id'] == participant_id]

                # Get metric values for this participant
                metric1_value = participant_data[metric1_name].values[0]
                metric2_value = participant_data[metric2_name].values[0]

                # Hierarchical grouping:
                # Group 0: metric1 <= threshold1 (regardless of metric2) OR metric1 is NaN
                # Group 1: metric1 > threshold1 AND metric2 <= threshold2
                # Group 2: metric1 > threshold1 AND metric2 > threshold2
                # If any metric is NaN, assign to group 0
                if pd.isna(metric1_value) or metric1_value <= threshold1:
                    group_idx = 0
                elif pd.isna(metric2_value) or metric2_value <= threshold2:
                    group_idx = 1
                else:
                    group_idx = 2

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

            # Generate labels and plot for hierarchical case
            plot_trajectory_groups(ax, group_data, time_points, time_point_mapping,
                                 score, [metric1, metric2], group_colors, num_groups,
                                 [(threshold1, threshold2)], single_metric=False)

        # Save the plot
        plt.tight_layout()
        num_subjects = len(df)
        if len(metric_names) == 1:
            figure_fname = os.path.join(output_dir,
                                        f'trajectory_plot_{score}_{metric_names[0]}_{num_subjects}subjects.png')
            print(f'Trajectory plot for {score} by {metric_names[0]} saved as {figure_fname}')
        else:
            figure_fname = os.path.join(output_dir,
                                        f'trajectory_plot_{score}_{metric_names[0]}_{metric_names[1]}_{num_subjects}subjects.png')
            print(f'Trajectory plot for {score} by {metric_names[0]} and {metric_names[1]} saved as {figure_fname}')
        plt.savefig(figure_fname, dpi=300)
        plt.close()


def plot_trajectory_groups(ax, group_data, time_points, time_point_mapping,
                          score, metrics, group_colors, num_groups, thresholds, single_metric=True):
    """
    Helper function to plot trajectory groups with proper labels and styling.

    :param ax: matplotlib axes object
    :param group_data: list of dictionaries containing group data for each time point
    :param time_points: list of time points
    :param time_point_mapping: dict mapping time points to months
    :param score: clinical score name
    :param metrics: metric name (single) or list of metric names (hierarchical)
    :param group_colors: list of colors for each group
    :param num_groups: number of groups
    :param thresholds: list of tuples containing threshold values
    :param single_metric: boolean indicating if this is a single metric case
    """

    # Get the actual month values for x-axis positioning
    time_points_months = [time_point_mapping[tp]['months'] for tp in time_points]
    # Find the last time point with data for counting subjects in legend
    last_time_point = time_points[-1]  # Use the last time point in the list
    # For custom legend
    group_labels = [""] * num_groups

    # Calculate and plot mean ± confidence interval (CI) trajectories for each group
    for tp in time_points:
        # Get x position in months
        tp_month = time_point_mapping[tp]['months']

        for group_idx in range(num_groups):
            # Get group values for this time point for given clinical score (e.g., 'lems_bl', 'lems_1m', etc.)
            group_values = group_data[group_idx][tp]

            if group_values:
                # Ensure all values are numeric
                group_values = [float(val) for val in group_values]
                group_mean = np.mean(group_values)      # Mean value for given clinical score at this time point
                # Calculate 95% confidence interval
                group_ci = 1.96 * np.std(group_values) / np.sqrt(len(group_values)) if len(group_values) > 1 else 0

                # Create group label based on whether it's single metric or hierarchical
                # Count subjects at the last time point for legend
                if single_metric:
                    metric = metrics
                    threshold = thresholds[0][0]
                    unit = '%' if 'ratio' in metric else 'mm'
                    last_tp_count = len(group_data[group_idx][last_time_point])
                    if group_idx == 0:
                        group_labels[group_idx] = f'{METRIC_TO_TITLE[metric]} ≤{threshold:.2f} {unit} (n={last_tp_count})'
                    else:
                        group_labels[group_idx] = f'{METRIC_TO_TITLE[metric]} >{threshold:.2f} {unit} (n={last_tp_count})'
                else:
                    # Hierarchical case (two metrics: lesion width --> total tissue bridge)
                    metric1, metric2 = metrics[0], metrics[1]
                    threshold1, threshold2 = thresholds[0][0], thresholds[0][1]
                    unit1 = '%' if 'ratio' in metric1 else 'mm'
                    unit2 = '%' if 'ratio' in metric2 else 'mm'
                    last_tp_count = len(group_data[group_idx][last_time_point])

                    if group_idx == 0:
                        group_labels[group_idx] = f'{METRIC_TO_TITLE[metric1]} ≤{threshold1:.2f} {unit1} (n={last_tp_count})'
                    elif group_idx == 1:
                        group_labels[group_idx] = f'{METRIC_TO_TITLE[metric1]} >{threshold1:.2f} {unit1} & {METRIC_TO_TITLE[metric2]} ≤{threshold2:.2f} {unit2} (n={last_tp_count})'
                    else:
                        group_labels[group_idx] = f'{METRIC_TO_TITLE[metric1]} >{threshold1:.2f} {unit1} & {METRIC_TO_TITLE[metric2]} >{threshold2:.2f} {unit2} (n={last_tp_count})'

                print(f'{score}, {metrics}, Time Point {tp} ({tp_month}m): Mean {score}={group_mean:.2f}, N={len(group_values)}')

                # Only show label in legend for the first actual time point that has data
                ax.errorbar(tp_month, group_mean, yerr=group_ci,
                            fmt='o', color=group_colors[group_idx], ecolor=group_colors[group_idx],
                            markersize=8, capsize=8,
                            capthick=2.5,       # linewidth of the caps only
                            elinewidth=2.5,     # linewidth of the vertical errorbar line
                            label=None)

    # Connect mean points with lines for each group
    for group_idx in range(num_groups):
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

    # # Set labels and title
    # if single_metric:
    #     title_metric = METRIC_TO_TITLE[metrics]
    # else:
    #     title_metric = f'{METRIC_TO_TITLE[metrics[0]]} & {METRIC_TO_TITLE[metrics[1]]}'
    #
    # ax.set_title(f'{CLINICAL_SCORES_TO_AXES[score]} over time stratified by {title_metric}', fontsize=FONT_SIZE)
    ax.set_xlabel('Time Point', fontsize=FONT_SIZE)
    ax.set_ylabel(f'{CLINICAL_SCORES_TO_AXES[score]}', fontsize=FONT_SIZE)

    # Set x-ticks at the actual month points (0, 1, 3, 6)
    ax.set_xticks(time_points_months)
    ax.set_xticklabels(['BL', 'M1', 'M3', 'M6'], fontsize=FONT_SIZE)

    # Add minor ticks only at the labeled tick positions for better visualization
    ax.tick_params(axis='x', which='minor', bottom=True, length=4)
    # Place minor ticks at the same positions as the major ticks
    ax.set_xticks(time_points_months, minor=True)
    # Set x and y-tick font size
    ax.tick_params(axis='y', labelsize=FONT_SIZE)
    ax.tick_params(axis='x', labelsize=FONT_SIZE)

    # Custom legend handles: horizontal line + circle marker
    legend_handles = [
        Line2D([0], [0], color=group_colors[i], marker='o', linestyle='-',
               linewidth=2.5, markersize=8)
        for i in range(num_groups)
        if group_labels[i]
    ]
    legend_labels = [group_labels[i] for i in range(num_groups) if group_labels[i]]
    ax.legend(legend_handles, legend_labels, loc='upper left',
              fontsize=FONT_SIZE - 6, framealpha=0.9)

    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)



def main():

    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Create output directory
    output_dir = args.o
    os.makedirs(output_dir, exist_ok=True)

    # Read the data files
    print("\nReading data files...")
    df = read_csv_file_with_lesion_metrics(args.i)
    # Keep only relevant columns: participant_id, midsagittal_length_sct, midsagittal_width_sct, total_tissue_bridge_sct
    # NOTE: although this file also contains clinical scores, we will read clinical data from separate files
    df = df[['participant_id',
             'session_id',
             'midsagittal_length_sct',
             'midsagittal_width_sct',
             'total_tissue_bridge_sct']]

    # --------------
    # sci-zurich clinical data
    # --------------
    df_clinical_sci_zurich = pd.read_excel(args.file_clinical_sci_zurich, engine='openpyxl',
                                         usecols=['participant_id', 'session_id',
                                                  'lems_bl', 'lems_1m', 'lems_3m', 'lems_6m',
                                                  'ms_bl', 'ms_1m', 'ms_3m', 'ms_6m',
                                                  'pp_bl', 'pp_1m', 'pp_3m', 'pp_6m',
                                                  'lt_bl', 'lt_1m', 'lt_3m', 'lt_6m'])
    # If session_id is empty, fill with 'ses-01'
    df_clinical_sci_zurich['session_id'] = df_clinical_sci_zurich['session_id'].fillna('ses-01')
    # Replace 'NT' with NaN
    df_clinical_sci_zurich = df_clinical_sci_zurich.replace('NT', np.nan)
    # Merge the dataframes
    print("\nMerging dataframes...")
    df = pd.merge(df, df_clinical_sci_zurich, on=['participant_id', 'session_id'], how='left')

    # -------------
    # NISCI clinical data
    # -------------
    df_clinical_nisci = pd.read_excel(args.file_clinical_nisci, engine='openpyxl',
                                      usecols=['Patient',
                                               'LEMS_01', 'LEMS_03', 'LEMS_05', 'LEMS_06',
                                               'TMS_01', 'TMS_03', 'TMS_05', 'TMS_06',
                                               'TPP_01', 'TPP_03', 'TPP_05', 'TPP_06',
                                               'TLT_01', 'TLT_03','TLT_05', 'TLT_06'])
    # '01' -- Day 0 (Screening)
    # '02' -- Day 1 (Baseline)
    # '03' -- 2 Weeks (14 days)
    # '04' -- 1 month (30 days)
    # '05' -- 3 months (84 days)
    # '06' -- 6 months (168 days)
    # NOTE: using '03' as 1 moth time; for details see nisci-trial/README.md
    df_clinical_nisci = df_clinical_nisci.rename(columns={'Patient': 'participant_id'})
    # Rename columns to lowercase and consistent naming with sci-zurich
    df_clinical_nisci = df_clinical_nisci.rename(columns={
        'LEMS_01': 'lems_bl',
        'LEMS_03': 'lems_1m',
        'LEMS_05': 'lems_3m',
        'LEMS_06': 'lems_6m',
        'TMS_01': 'ms_bl',
        'TMS_03': 'ms_1m',
        'TMS_05': 'ms_3m',
        'TMS_06': 'ms_6m',
        'TPP_01': 'pp_bl',
        'TPP_03': 'pp_1m',
        'TPP_05': 'pp_3m',
        'TPP_06': 'pp_6m',
        'TLT_01': 'lt_bl',
        'TLT_03': 'lt_1m',
        'TLT_05': 'lt_3m',
        'TLT_06': 'lt_6m'
    })

    # NOTE: these columns already exist in df from sci-zurich, so we only add missing values from nisci
    df = pd.merge(df, df_clinical_nisci, on='participant_id', how='left', suffixes=('', '_nisci'))
    for score in CLINICAL_SCORES_TO_AXES.keys():
        for tp in ['bl', '1m', '3m', '6m']:
            col = f'{score}_{tp}'
            col_nisci = f'{col}_nisci'
            if col in df.columns and col_nisci in df.columns:
                # Fill missing values in df[col] with values from df[col_nisci]
                df[col] = df[col].combine_first(df[col_nisci])
                # Drop the nisci column
                df = df.drop(columns=[col_nisci])

    # Drop subjects with missing 6-month clinical scores
    print(f'Number of subjects before dropping missing 6-month data: {df.shape[0]}')
    df = df[~df['lems_6m'].isna() & ~df['ms_6m'].isna() & ~df['pp_6m'].isna() & ~df['lt_6m'].isna()]
    print(f'Number of subjects after dropping missing 6-month data: {df.shape[0]}')

    #----------------
    # Clinical scores and baseline metrics over time
    #----------------
    create_trajectory_plots(df, output_dir)


if __name__ == '__main__':
    main()
