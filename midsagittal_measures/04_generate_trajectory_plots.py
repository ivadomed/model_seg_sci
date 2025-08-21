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

from sklearn.cluster import KMeans


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
    Compute normalized recovery rates by calculating the change from baseline (or the first available measurement) to
    follow-up and dividing them by the maximal score improvable.
    Then, scale the normalized sensorimotor recovery rates using min-max scaling method (per participant and score):
        x' = (x - min(x)) / (max(x) - min(x))

    For each subject:
        1. Identifies the first available time point (baseline or 1m)
        2. Computes maximal improvable score from this first time point
        3. Normalizes subsequent scores by dividing improvement by maximal improvable score
        4. Applies min-max scaling to the normalized recovery rates per participant and score

    :param df: pandas DataFrame with baseline lesion metrics and clinical scores across
    multiple time points
    :return df: pandas DataFrame with min-max scaled normalized clinical scores
    """
    # Get unique participant IDs
    participants = df['participant_id'].unique()

    # Time points in order
    time_points = ['bl', '1m', '3m', '6m', '12m']

    # First pass: compute normalized recovery rates by calculating the change from baseline (i.e., first available
    # measurement) to follow-up and dividing them by the maximal score improvable
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

    # Second pass: apply min-max scaling to normalized recovery rates (separately for each participant and
    # clinical score)
    normalized_columns = [col for col in df.columns if col.endswith('_improvement_normalized')]

    for participant in participants:
        participant_idx = df['participant_id'] == participant
        participant_data = df.loc[participant_idx]

        # Process each clinical score separately
        for score in CLINICAL_SCORES_TO_AXES.keys():
            # Get normalized values for this participant and this specific clinical score
            score_normalized_values = []
            score_columns = []

            # Debug - subject with decreased light touch score for 1m from baseline
            # if participant == 'sub-zh12' and score == 'lt':
            #     print('here')

            for col in normalized_columns:
                if col.startswith(f"{score}_") and col in participant_data.columns and not pd.isna(participant_data[col].values[0]):
                    score_normalized_values.append(participant_data[col].values[0])
                    score_columns.append(col)

            # Apply min-max scaling if there are values for this participant and this score
            if len(score_normalized_values) > 0:
                min_val = min(score_normalized_values)
                max_val = max(score_normalized_values)

                # Apply min-max scaling: x' = (x - min(x)) / (max(x) - min(x))
                if max_val != min_val:  # Avoid division by zero
                    for col in score_columns:
                        original_val = participant_data[col].values[0]
                        scaled_val = (original_val - min_val) / (max_val - min_val)
                        # Create new column with _scaled suffix
                        scaled_col = col.replace('_improvement_normalized', '_improvement_normalized_scaled')
                        df.loc[participant_idx, scaled_col] = scaled_val
                else:
                    # If all values are the same for this participant and score, set to 0.5 (middle value)
                    for col in score_columns:
                        # Create new column with _scaled suffix
                        scaled_col = col.replace('_improvement_normalized', '_improvement_normalized_scaled')
                        df.loc[participant_idx, scaled_col] = 0.5

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


def create_trajectory_plots(df, output_dir, method, stratification_method='kmeans', n_groups=3):
    """
    Create an individual trajectory plot showing clinical scores across time points for each participant,
    with lines colored by baseline lesion metrics.
    Also creates plots showing normalized improvement scores for follow-up time points.

    :param df: pandas dataframe with baseline lesion metrics and clinical scores across
    multiple time points
    :param output_dir: output directory
    :param method: str: method ('GT' or 'SCIsegV2')
    :param stratification_method: str: method for computing thresholds ('fixed', 'kmeans')
    :param n_groups: int: number of groups to create (default: 3)
    """
    # Set font to Arial
    plt.rcParams['font.sans-serif'] = 'Arial'

    # Compute metric thresholds based on the chosen stratification method
    if stratification_method == 'fixed':
        # Use predefined thresholds
        metric_thresholds = {
            'midsagittal_length': [0, 10, 20],
            'midsagittal_width': [0, 3, 6],
            'ventral_tissue_bridge': [0, 1],
            'dorsal_tissue_bridge': [0, 1],
            'total_tissue_bridge': [0, 1, 2],
            'dorsal_bridge_ratio': [0, 50],
            'ventral_bridge_ratio': [0, 50]
        }
    else:
        # Compute thresholds using K-means clustering
        metric_thresholds = {}
        for metric in METRIC_TO_TITLE.keys():
            thresholds = compute_kmeans_thresholds(df, metric, n_groups, visualize=True, output_dir=output_dir)

            metric_thresholds[metric] = thresholds
            print(f"K-means thresholds for {metric} ({stratification_method}): {[f'{t:.2f}' for t in thresholds]}")

    # Define colors for each group (adjust based on number of groups)
    if n_groups <= 3:
        group_colors = ['blue', 'green', 'red'][:n_groups]
    else:
        # Generate more colors using matplotlib colormap
        import matplotlib.cm as cm
        group_colors = [cm.Set1(i/n_groups) for i in range(n_groups)]

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

    # PART 3: Normalized and scaled improvement scores trajectory plots
    create_normalized_scaled_trajectory_plots(df, group_colors, method, metric_thresholds, output_dir,
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
            #   1. 'GT' - manual lesion masks + sct_analyze_lesion --> '_manual' suffix
            #   2. 'SCIsegV2' - SCIsegV2 lesion masks + sct_analyze_lesion --> '_sct' suffix
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
                         f'{METRIC_TO_TITLE[metric].split("[")[0]}', fontsize=FONT_SIZE)
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
                         f'{METRIC_TO_TITLE[metric].split("[")[0]}', fontsize=FONT_SIZE)
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


def create_normalized_scaled_trajectory_plots(df, group_colors, method, metric_thresholds, output_dir,
                                       time_point_mapping, time_points):
    """
    Create trajectory plots for normalized and scaled clinical scores across follow-up time points for each participant,
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
                    normalized_col = f"{score}_{tp}_improvement_normalized_scaled"

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
            ax.set_title(f'Normalized scaled improvement in {CLINICAL_SCORES_TO_AXES[score]} stratified by '
                         f'{METRIC_TO_TITLE[metric].split("[")[0]}', fontsize=FONT_SIZE)
            ax.set_xlabel('Time Point', fontsize=FONT_SIZE)
            ax.set_ylabel(f'Normalized Scaled Improvement', fontsize=FONT_SIZE)

            # Add a horizontal line at y=0 (no improvement)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

            # ax.set_ylim(-0.2, 1.2)

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
                                        f'{METHOD_TO_FNAME[method]}_normalized_scaled_improvement_{score}_{metric}_{num_subjects}subjects.png')
            plt.savefig(figure_fname, dpi=300)
            print(f'Normalized scaled improvement plot for {score} by {metric} saved as {figure_fname}')
            plt.close()
        # Combine individual trajectory plots for this score
        combine_plot(f'{METHOD_TO_FNAME[method]}_normalized_scaled_improvement_{score}', num_subjects, output_dir)


def compute_kmeans_thresholds(df, metric, n_groups=3, visualize=True, output_dir=None):
    """
    Use K-means clustering to determine optimal thresholds for stratification.

    :param df: pandas DataFrame with lesion metrics
    :param metric: str: metric name
    :param n_groups: int: number of groups to create
    :param visualize: bool: whether to create visualization plots
    :param output_dir: str: directory to save visualization plots
    :return: list of threshold values
    """
    metric_name = f'{metric}_sct'
    values = df[metric_name].dropna().values.reshape(-1, 1)

    # Fit K-means
    kmeans = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
    labels = kmeans.fit_predict(values)

    # Get cluster centers and their original indices, then sort by center value
    centers_with_indices = [(i, center[0]) for i, center in enumerate(kmeans.cluster_centers_)]
    centers_with_indices.sort(key=lambda x: x[1])  # Sort by center value

    # Extract sorted centers and create mapping from original label to sorted index
    centers = [center for _, center in centers_with_indices]
    original_to_sorted = {orig_idx: sorted_idx for sorted_idx, (orig_idx, _) in enumerate(centers_with_indices)}

    # Create thresholds at midpoints between centers
    thresholds = [values.min()]
    for i in range(len(centers) - 1):
        threshold = (centers[i] + centers[i + 1]) / 2
        thresholds.append(threshold)
    thresholds.append(values.max() + 0.01)  # slight offset for inclusive upper bound

    # Print group information
    print(f"\nK-means clustering results for {metric}:")
    print(f"Number of groups: {n_groups}")
    print(f"Cluster centers: {[f'{c:.2f}' for c in centers]}")
    print(f"Thresholds: {[f'{t:.2f}' for t in thresholds]}")

    # Count subjects in each group
    for i in range(n_groups):
        cluster_mask = labels == i
        n_subjects = np.sum(cluster_mask)
        # Use the sorted index for consistent coloring (because K-means algorithm assigns cluster labels (0, 1, etc.)
        # based on the order it finds clusters during fitting
        sorted_idx = original_to_sorted[i]
        cluster_center = centers[sorted_idx]

        # Determine group range
        if sorted_idx == 0:
            range_text = f"< {thresholds[1]:.2f}"
        elif sorted_idx == n_groups - 1:
            range_text = f"≥ {thresholds[sorted_idx]:.2f}"
        else:
            range_text = f"{thresholds[sorted_idx]:.2f} - {thresholds[sorted_idx+1]:.2f}"

        unit = '%' if 'ratio' in metric else 'mm'
        print(f"Group {sorted_idx+1}: {range_text} {unit} (center: {cluster_center:.2f}, n={n_subjects})")

    # Create visualization if requested
    if visualize and output_dir:
        # Set up the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot 1: Histogram with K-means results
        ax1.hist(values.flatten(), bins=30, alpha=0.7, color='lightblue', edgecolor='black')

        # Color points by cluster assignment
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray'][:n_groups]

        # Add cluster centers
        for i, center in enumerate(centers):
            ax1.axvline(center, color=colors[i], linestyle='--', linewidth=2,
                       label=f'Cluster {i+1} center: {center:.2f}')

        # Add thresholds
        for i, threshold in enumerate(thresholds[1:-1], 1):  # Skip first and last
            ax1.axvline(threshold, color='black', linestyle='-', linewidth=1.5, alpha=0.8,
                       label=f'Threshold {i}: {threshold:.2f}')

        # Compute and plot median line
        median_value = np.median(values)
        ax1.axvline(median_value, color='gray', linestyle=':', linewidth=2, label='Median')

        ax1.set_xlabel(f'{METRIC_TO_TITLE[metric]}', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'K-means Clustering for {METRIC_TO_TITLE[metric]} (n_groups={n_groups})', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Scatter plot showing cluster assignments
        y_jitter = np.random.normal(0, 0.05, len(values))  # Add jitter for better visualization

        for i in range(n_groups):
            cluster_mask = labels == i
            cluster_values = values[cluster_mask]
            cluster_jitter = y_jitter[cluster_mask]

            # Use the sorted index for consistent coloring (because K-means algorithm assigns cluster labels (0, 1,
            # etc.) based on the order it finds clusters during fitting
            sorted_idx = original_to_sorted[i]
            ax2.scatter(cluster_values, cluster_jitter, c=colors[sorted_idx], alpha=0.6, s=50,
                       label=f'Group {sorted_idx+1} (n={np.sum(cluster_mask)})')

        # Add cluster centers
        for i, center in enumerate(centers):
            ax2.axvline(center, color=colors[i], linestyle='--', linewidth=2, alpha=0.8)

        # Add thresholds
        for threshold in thresholds[1:-1]:  # Skip first and last
            ax2.axvline(threshold, color='black', linestyle='-', linewidth=1.5, alpha=0.8,
                        label=f'Threshold {i}: {threshold:.2f}')

        # Plot median line
        ax2.axvline(median_value, color='gray', linestyle=':', linewidth=2, label='Median')

        ax2.set_xlabel(f'{METRIC_TO_TITLE[metric]}', fontsize=12)
        ax2.set_ylabel('Random Jitter (for visualization)', fontsize=12)
        ax2.set_title(f'Subject Assignment to Groups', fontsize=14)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.5, 0.5)

        plt.tight_layout()

        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = os.path.join(output_dir, f'kmeans_{n_groups}_groups_{metric}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"K-means visualization saved as: {plot_filename}")

    return thresholds


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


    output_dir = args.o
    os.makedirs(output_dir, exist_ok=True)
    #----------------
    # Clinical scores and baseline metrics over time
    #----------------
    create_trajectory_plots(df, output_dir, method)


if __name__ == '__main__':
    main()
