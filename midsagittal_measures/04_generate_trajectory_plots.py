"""
Figure 5.

Generate trajectory plots of clinical scores over time. Participants are grouped based on cutoffs of baseline lesion
metrics obtained from URP-CTREE analysis.

Generate 2x2 figure showing trajectory plots for:
- LEMS
- Total Motor Score
- Pinprick Score
- Light-Touch Score

The script generates two types of plots:
1. Raw trajectory plots: Mean ± 95% CI trajectories with time points shown as months (0, 1, 3, 6)
2. LME trajectory plots: Linear Mixed-Effects (LME) fitted trajectories using actual days from baseline
   - Individual trajectories shown as thin, semi-transparent lines
   - Population-level LME fitted lines shown as thick, solid lines
   - Uses actual timepoint data (days) from the -file-timepoints XLSX file
   - LME model: score ~ days (fixed) + (1 + days | participant_id) (random intercept and slope)
3. LME trajectory plots with log(days + 1) transformation to capture non-linear recovery patterns
   - Individual trajectories shown as thin, semi-transparent lines
   - Population-level LME fitted lines shown as thick, solid lines
   - Uses actual timepoint data (days) from the -file-timepoints XLSX file
   - NLME model: score ~ log(days + 1) * C(group_idx) (fixed) + (1 + log(days + 1) | participant_id) (random intercept and slope)
   - Log transformation captures non-linear recovery patterns typical in clinical data (rapid early recovery that plateaus)

The script:
- reads CSV file with lesion metrics (Note: we don't use the lesion metrics but need participant_id column)
- reads XLSX files with clinical scores for both datasets
- reads XLSX file with exact timepoints in days for clinical assessments (optional, uses approximate days if not provided)
- merges the data into a single dataframe
- creates trajectory plots of clinical scores over time

Example usage:
    python 04_generate_trajectory_plots.py \
        -i <PATH_TO_CSV_FILE> \
        -file-clinical-nisci <PATH_TO_CLINICAL_SCORES_XLSX_NISCI> \
        -file-clinical-sci-zurich <PATH_TO_CLINICAL_SCORES_XLSX_SCI_ZURICH> \
        -file-timepoints <PATH_TO_TIMEPOINTS_XLSX> \
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
import warnings

from matplotlib.lines import Line2D
from statsmodels.formula.api import mixedlm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from utils import read_csv_file_with_lesion_metrics

METRIC_TO_TITLE = {
    'midsagittal_length': 'Lesion Length',
    'midsagittal_width': 'Lesion Width',
    'total_tissue_bridge': 'Total Tissue Bridges'
}

CLINICAL_SCORES_TO_AXES = {
    'uems': 'UEMS',
    'lems': 'Lower Extremity Motor Score',
    'ms': 'Total Motor Score',
    'pp': 'Pinprick Score',
    'lt': 'Light Touch Score'
}

SCORE_TO_YLIM = {
    'uems': (0, 50),
    'lems': (-3, 60),
    'ms': (5, 115),
    'pp': (8, 80),
    'lt': (15, 100)
}


FONT_SIZE = 20


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='Generate trajectory plots of clinical scores over time.',
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
        '-file-timepoints',
        required=False,
        type=str,
        help='Absolute path to a XLSX file with exact timepoints for clinical assessments for both datasets.'
    )
    parser.add_argument(
        '-o',
        required=True,
        type=str,
        help='Path to the output folder where figures and tables will be saved.'
    )

    return parser



def combine_plots(figure_fnames, output_type, output_dir):
    """
    Combine all the plots into a single figure using bash convert command
    This requires ImageMagick to be installed
    :param figure_fnames: list of figure filenames
    :param output_type: str: type of figure (e.g., 'lme' or 'raw') to fname
    :param output_dir: str: output directory where the combined figure will be saved
    """

    # Create 'combined' directory if it does not exist
    combined_dir = os.path.join(output_dir, 'combined')
    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)

    # 2 x 2:
    # lems x ms
    # pp x lt
    figure_types = ['lems', 'ms', 'pp', 'lt']

    # Create a dictionary to map figure types to their file paths
    figure_dict = {}
    for fname in figure_fnames:
        for fig_type in figure_types:
            if fig_type in fname:
                figure_dict[fig_type] = fname
                break

    # Create labeled copies with panel labels
    labeled_files = []
    panel_labels = ['A)', '', 'B)', '']  # A for first row first figure (lems), B for second row first figure (pp)

    for i, fig_type in enumerate(figure_types):
        if fig_type in figure_dict:
            input_file = figure_dict[fig_type]
            labeled_file = os.path.join(combined_dir, f'{fig_type}_labeled.png')
            panel_label = panel_labels[i]

            # Add panel label using ImageMagick convert (only if label is not empty)
            if panel_label:
                # Add white space above figure and then add label
                cmd = f"convert '{input_file}' -bordercolor white -border 0x80+0+0 -pointsize {FONT_SIZE*5} -fill black -gravity NorthWest -annotate +20+20 '{panel_label}' '{labeled_file}'"
            else:
                # Add same white space above figure without label for consistency
                cmd = f"convert '{input_file}' -bordercolor white -border 0x80+0+0 '{labeled_file}'"
            subprocess.run(cmd, shell=True, check=True)
            labeled_files.append(labeled_file)

    if len(labeled_files) == 4:
        # Combine into 2x2 grid
        output_combined = os.path.join(combined_dir, f'Fig5_trajectory_plots_{output_type}_combined.png')
        cmd = f"convert '{labeled_files[0]}' '{labeled_files[1]}' +append temp_row1.png && " \
              f"convert '{labeled_files[2]}' '{labeled_files[3]}' +append temp_row2.png && " \
              f"convert temp_row1.png temp_row2.png -append '{output_combined}' && " \
              f"rm temp_row1.png temp_row2.png"

        subprocess.run(cmd, shell=True, check=True)
        print(f"Combined trajectory plots saved as: {output_combined}")

        # Also save as TIFF at 300 DPI
        output_combined_tiff = output_combined.replace('.png', '.tiff')
        cmd_tiff = f"convert '{output_combined}' -density 300 -compress lzw '{output_combined_tiff}'"
        subprocess.run(cmd_tiff, shell=True, check=True)
        print(f"Combined trajectory plots also saved as: {output_combined_tiff}")
    else:
        print(f"Warning: Expected 4 figures but found {len(labeled_files)}. Cannot create 2x2 grid.")



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

    # group_colors = ['green', 'red', 'blue']
    group_colors = [
        '#009E73',  # green
        '#E34A33',  # red
        '#E69F00',  # orange
    ]

    # Define time points with their actual time values in months from baseline
    time_point_mapping = {
        'bl': {'order': 0, 'months': 0},     # baseline = 0 months
        '1m': {'order': 1, 'months': 1},     # 1 month
        '3m': {'order': 2, 'months': 3},     # 3 months
        '6m': {'order': 3, 'months': 6},     # 6 months
    }
    # Sort time points in a logical order
    time_points = sorted(list(time_point_mapping.keys()), key=lambda x: time_point_mapping.get(x, {}).get('order'))

    # PART 1: Raw trajectory plots (original, using months)
    create_raw_trajectory_plots(df, group_colors, metric_thresholds, output_dir, time_point_mapping, time_points)

    # PART 2: LME-fitted trajectory plots (using actual days)
    print("\n" + "="*60)
    print("Generating LME-fitted trajectory plots with days on x-axis...")
    print("="*60)
    create_lme_trajectory_plots(df, group_colors, metric_thresholds, output_dir)

    # PART 3: NLME-fitted trajectory plots (using actual days)
    print("\n" + "="*60)
    print("Generating Non-Linear Mixed-Effects (NLME) trajectory plots with days on x-axis...")
    print("Using log(days + 1) transformation to capture non-linear recovery patterns")
    print("="*60)
    create_nlme_trajectory_plots(df, group_colors, metric_thresholds, output_dir)



def reshape_to_long_format(df, score):
    """
    Reshape dataframe from wide to long format for LME modeling.

    :param df: pandas dataframe in wide format with columns like lems_bl, lems_1m, etc.
    :param score: clinical score name (e.g., 'lems', 'ms')
    :return: dataframe in long format with columns: participant_id, days, score_value, group_idx, metric columns
    """
    time_points = ['bl', '1m', '3m', '6m']

    # Create list to store reshaped data
    long_data = []

    for _, row in df.iterrows():
        participant_id = row['participant_id']

        for tp in time_points:
            # Get clinical score value
            score_col = f'{score}_{tp}'
            if score_col in df.columns and not pd.isna(row[score_col]):
                score_value = float(row[score_col])

                # Get days for this timepoint (if available)
                days_col = tp
                if days_col in df.columns and not pd.isna(row[days_col]):
                    days = float(row[days_col])
                else:
                    # Fallback to approximate days if exact timepoint not available
                    days_mapping = {'bl': 0, '1m': 30, '3m': 90, '6m': 180}
                    days = days_mapping[tp]

                # Create row for long format
                long_row = {
                    'participant_id': participant_id,
                    'timepoint': tp,
                    'days': days,
                    'score_value': score_value,
                }

                # Add metric columns
                for col in df.columns:
                    if col.endswith('_sct') or col == 'session_id':
                        long_row[col] = row[col]

                long_data.append(long_row)

    return pd.DataFrame(long_data)


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

    figure_fnames = []
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
        figure_fnames.append(figure_fname)
        plt.savefig(figure_fname, dpi=300)
        plt.close()

    combine_plots(figure_fnames, 'raw', output_dir)


def create_lme_trajectory_plots(df, group_colors, metric_thresholds, output_dir):
    """
    Create trajectory plots with Linear Mixed-Effects (LME) fitted lines using actual days on x-axis.

    Fits a single LME model with group interaction: score ~ days * C(group_idx) + (1 + days | participant_id)
        - C(group_idx) treats group_idx as a categorical variable, allowing for different intercepts and slopes for each group (e.g., lesion width < XX mm and lesion width > XX mm).
        - The interaction term (days * C(group_idx)) allows the slope of days to differ between groups.
        - (1 + days | participant_id) allows for random intercepts and slopes for each participant, accounting for individual variability in trajectories.

    :param df: pandas dataframe with baseline lesion metrics and clinical scores
    :param group_colors: list of colors for each group
    :param metric_thresholds: dict: thresholds for each metric to create groups
    :param output_dir: output directory
    """

    figure_fnames = []

    # Loop over clinical scores (e.g., 'lems', 'ms', ...)
    for score, metrics in metric_thresholds.items():
        metric_names = list(metrics.keys())

        # Reshape data to long format for LME
        df_long = reshape_to_long_format(df, score)

        if df_long.empty:
            print(f"No data available for {score}, skipping...")
            continue

        # Single metric - 2 groups
        if len(metric_names) == 1:
            metric = metric_names[0]
            metric_name = f'{metric}_sct'
            threshold = metric_thresholds[score][metric][0]

            # Assign groups
            df_long['group_idx'] = df_long[metric_name].apply(
                lambda x: 0 if pd.isna(x) or x <= threshold else 1
            )
            num_groups = 2

        # Multiple metrics - 3 groups (hierarchical)
        else:
            metric1, metric2 = metric_names[0], metric_names[1]
            metric1_name = f'{metric1}_sct'
            metric2_name = f'{metric2}_sct'
            threshold1 = metric_thresholds[score][metric1][0]
            threshold2 = metric_thresholds[score][metric2][0]

            def assign_group(row):
                m1_val = row[metric1_name]
                m2_val = row[metric2_name]
                if pd.isna(m1_val) or m1_val <= threshold1:
                    return 0
                elif pd.isna(m2_val) or m2_val <= threshold2:
                    return 1
                else:
                    return 2

            df_long['group_idx'] = df_long.apply(assign_group, axis=1)
            num_groups = 3

        # Convert group_idx to categorical for proper interaction term handling
        df_long['group_idx'] = df_long['group_idx'].astype('category')

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot individual trajectories (light, thin lines) and fit LME for each group
        group_labels = [""] * num_groups

        # Fit a Linear Mixed-Effects Model
        # C(group_idx) forces patsy/statsmodels to treat group_idx as a *categorical*
        # predictor, even though it is encoded as integers (0/1/2). Without C(...), the model
        # would interpret group_idx as a continuous numeric variable and (incorrectly) assume that
        # going from group 0 -> 1 -> 2 has a linear effect. With C(group_idx), each group gets its
        # own intercept (and its own slope when interacted with time), which is what we want here.
        y_col = 'score_value'
        x_col = 'days'
        formula = f"{y_col} ~ {x_col} * C(group_idx)"
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                lme_model = mixedlm(formula, df_long,
                                   groups=df_long['participant_id'],
                                   re_formula='~days')
                lme_result = lme_model.fit(method='lbfgs', maxiter=200)
        except Exception as e:
            lme_result = None
            print(f"Warning: LME fitting failed for {score}: {str(e)}")

        for group_idx in range(num_groups):
            group_data = df_long[df_long['group_idx'] == group_idx].copy()

            if group_data.empty:
                continue

            # Count unique subjects at last timepoint (6m)
            last_tp_subjects = group_data[group_data['timepoint'] == '6m']['participant_id'].nunique()

            # Create group label
            if len(metric_names) == 1:
                metric = metric_names[0]
                threshold = metric_thresholds[score][metric][0]
                unit = '%' if 'ratio' in metric else 'mm'
                if group_idx == 0:
                    group_labels[group_idx] = f'{METRIC_TO_TITLE[metric]} ≤{threshold:.2f} {unit} (n = {last_tp_subjects})'
                else:
                    group_labels[group_idx] = f'{METRIC_TO_TITLE[metric]} >{threshold:.2f} {unit} (n = {last_tp_subjects})'
            else:
                metric1, metric2 = metric_names[0], metric_names[1]
                threshold1 = metric_thresholds[score][metric1][0]
                threshold2 = metric_thresholds[score][metric2][0]
                unit1 = '%' if 'ratio' in metric1 else 'mm'
                unit2 = '%' if 'ratio' in metric2 else 'mm'
                if group_idx == 0:
                    group_labels[group_idx] = f'{METRIC_TO_TITLE[metric1]} ≤{threshold1:.2f} {unit1} (n = {last_tp_subjects})'
                elif group_idx == 1:
                    group_labels[group_idx] = f'{METRIC_TO_TITLE[metric1]} >{threshold1:.2f} {unit1} & {METRIC_TO_TITLE[metric2]} ≤{threshold2:.2f} {unit2} (n = {last_tp_subjects})'
                else:
                    group_labels[group_idx] = f'{METRIC_TO_TITLE[metric1]} >{threshold1:.2f} {unit1} & {METRIC_TO_TITLE[metric2]} >{threshold2:.2f} {unit2} (n = {last_tp_subjects})'

            # Plot individual trajectories (thin, semi-transparent)
            for participant_id in group_data['participant_id'].unique():
                participant_data = group_data[group_data['participant_id'] == participant_id].sort_values('days')
                if len(participant_data) >= 2:
                    ax.plot(participant_data['days'], participant_data['score_value'],
                           color=group_colors[group_idx], alpha=0.15, linewidth=0.8, zorder=1)

            # Plot model-based fitted line from the single LME
            if lme_result is not None:
                days_range = np.linspace(group_data['days'].min(), group_data['days'].max(), 100)
                pred_df = pd.DataFrame({'days': days_range, 'group_idx': group_idx})
                # Ensure the categorical levels match the training data (important for patsy).
                pred_df['group_idx'] = pd.Categorical(
                    pred_df['group_idx'], categories=df_long['group_idx'].cat.categories
                )

                predictions = lme_result.predict(exog=pred_df)

                ax.plot(days_range, predictions,
                        color=group_colors[group_idx], linewidth=3, zorder=3,
                        label=group_labels[group_idx])

                # Optional: print per-group intercept/slope implied by the fixed effects.
                fe = lme_result.fe_params
                base_intercept = float(fe.get('Intercept', np.nan))
                base_slope = float(fe.get('days', np.nan))
                if group_idx == 0:
                    intercept = base_intercept
                    slope = base_slope
                else:
                    g_term = f"C(group_idx)[T.{group_idx}]"
                    # statsmodels may order interaction names as 'days:C(group_idx)[T.x]' or vice versa.
                    int_a = f"days:C(group_idx)[T.{group_idx}]"
                    int_b = f"C(group_idx)[T.{group_idx}]:days"
                    intercept = base_intercept + float(fe.get(g_term, 0.0))
                    slope = base_slope + float(fe.get(int_a, fe.get(int_b, 0.0)))

                print(f"\n{'='*60}")
                print(f"LME Model for {score.upper()}, Group {group_idx}: {group_labels[group_idx]}")
                print(f"{'='*60}")
                print(f"Fixed-effects implied intercept: {intercept:.3f}")
                print(f"Fixed-effects implied days slope: {slope:.4f}")
                print(f"Number of subjects: {group_data['participant_id'].nunique()}")
                print(f"Number of observations: {len(group_data)}")
            else:
                # Fallback: plot mean trajectory if model fitting failed.
                mean_data = group_data.groupby('days')['score_value'].mean().reset_index()
                ax.plot(mean_data['days'], mean_data['score_value'],
                        color=group_colors[group_idx], linewidth=3, zorder=3,
                        label=group_labels[group_idx])

        # Customize plot
        ax.set_xlabel('Days from Baseline', fontsize=FONT_SIZE)
        ax.set_ylabel(f'{CLINICAL_SCORES_TO_AXES[score]}', fontsize=FONT_SIZE)
        ax.tick_params(axis='both', labelsize=FONT_SIZE)

        # Reorder legend
        legend_order = [0, 2, 1] if num_groups == 3 else [0, 1]
        handles, labels = ax.get_legend_handles_labels()
        ordered_handles = [handles[i] for i in legend_order if i < len(handles)]
        ordered_labels = [labels[i] for i in legend_order if i < len(labels)]
        ax.legend(ordered_handles, ordered_labels, loc='upper left',
                 fontsize=FONT_SIZE - 4, framealpha=0.9)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set ylim
        ax.set_ylim(SCORE_TO_YLIM[score][0], SCORE_TO_YLIM[score][1])
        # Set xlim to 200 days (~6 months)
        ax.set_xlim(-5, 200)

        plt.tight_layout()

        # Save figure
        num_subjects = df['participant_id'].nunique()
        if len(metric_names) == 1:
            figure_fname = os.path.join(output_dir,
                                       f'trajectory_plot_LME_{score}_{metric_names[0]}_{num_subjects}subjects.png')
            print(f'\nLME trajectory plot for {score} by {metric_names[0]} saved as {figure_fname}')
        else:
            figure_fname = os.path.join(output_dir,
                                       f'trajectory_plot_LME_{score}_{metric_names[0]}_{metric_names[1]}_{num_subjects}subjects.png')
            print(f'\nLME trajectory plot for {score} by {metric_names[0]} and {metric_names[1]} saved as {figure_fname}')

        figure_fnames.append(figure_fname)
        plt.savefig(figure_fname, dpi=300)
        plt.close()

    # Combine plots
    combine_plots(figure_fnames, 'lme', output_dir)


def create_nlme_trajectory_plots(df, group_colors, metric_thresholds, output_dir):
    """
    Create trajectory plots with Non-Linear Mixed-Effects (NLME) fitted lines using actual days on x-axis.

    Fits a single mixed-effects model with non-linear time component and group interaction:
    score ~ np.log(days + 1) * C(group_idx) + (1 + np.log(days + 1) | participant_id)
        - Uses logarithmic transformation of time (log(days + 1)) to capture non-linear recovery patterns (common in clinical recovery data where rapid initial improvement plateaus over time)
        - C(group_idx) treats group_idx as a categorical variable, allowing for different intercepts and slopes for each group (e.g., lesion width < XX mm and lesion width > XX mm).
        - The interaction term allows the non-linear slope to differ between groups.
        - (1 + np.log(days + 1) | participant_id) allows for random intercepts and slopes for each participant, accounting for individual variability in trajectories.

    :param df: pandas dataframe with baseline lesion metrics and clinical scores
    :param group_colors: list of colors for each group
    :param metric_thresholds: dict: thresholds for each metric to create groups
    :param output_dir: output directory
    """

    figure_fnames = []

    # Loop over clinical scores (e.g., 'lems', 'ms', ...)
    for score, metrics in metric_thresholds.items():
        metric_names = list(metrics.keys())

        # Reshape data to long format for LME
        df_long = reshape_to_long_format(df, score)

        if df_long.empty:
            print(f"No data available for {score}, skipping...")
            continue

        # Single metric - 2 groups
        if len(metric_names) == 1:
            metric = metric_names[0]
            metric_name = f'{metric}_sct'
            threshold = metric_thresholds[score][metric][0]

            # Assign groups
            df_long['group_idx'] = df_long[metric_name].apply(
                lambda x: 0 if pd.isna(x) or x <= threshold else 1
            )
            num_groups = 2

        # Multiple metrics - 3 groups (hierarchical)
        else:
            metric1, metric2 = metric_names[0], metric_names[1]
            metric1_name = f'{metric1}_sct'
            metric2_name = f'{metric2}_sct'
            threshold1 = metric_thresholds[score][metric1][0]
            threshold2 = metric_thresholds[score][metric2][0]

            def assign_group(row):
                m1_val = row[metric1_name]
                m2_val = row[metric2_name]
                if pd.isna(m1_val) or m1_val <= threshold1:
                    return 0
                elif pd.isna(m2_val) or m2_val <= threshold2:
                    return 1
                else:
                    return 2

            df_long['group_idx'] = df_long.apply(assign_group, axis=1)
            num_groups = 3

        # Convert group_idx to categorical for proper interaction term handling
        df_long['group_idx'] = df_long['group_idx'].astype('category')

        # Add non-linear time transformation for modeling recovery curves
        # log(days + 1) captures the typical pattern of rapid initial recovery that plateaus over time
        df_long['log_days'] = np.log(df_long['days'] + 1)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot individual trajectories (light, thin lines) and fit LME for each group
        group_labels = [""] * num_groups

        # Fit a Non-Linear Mixed-Effects Model
        # C(group_idx) forces patsy/statsmodels to treat group_idx as a *categorical*
        # predictor, even though it is encoded as integers (0/1/2). Without C(...), the model
        # would interpret group_idx as a continuous numeric variable and (incorrectly) assume that
        # going from group 0 -> 1 -> 2 has a linear effect. With C(group_idx), each group gets its
        # own intercept (and its own slope when interacted with time), which is what we want here.
        # Using log(days + 1) instead of days to capture non-linear recovery patterns.
        y_col = 'score_value'
        x_col = 'log_days'
        formula = f"{y_col} ~ {x_col} * C(group_idx)"
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                lme_model = mixedlm(formula, df_long,
                                   groups=df_long['participant_id'],
                                   re_formula='~log_days')
                lme_result = lme_model.fit(method='lbfgs', maxiter=200)
        except Exception as e:
            lme_result = None
            print(f"Warning: Non-linear mixed-effects model fitting failed for {score}: {str(e)}")

        for group_idx in range(num_groups):
            group_data = df_long[df_long['group_idx'] == group_idx].copy()

            if group_data.empty:
                continue

            # Count unique subjects at last timepoint (6m)
            last_tp_subjects = group_data[group_data['timepoint'] == '6m']['participant_id'].nunique()

            # Create group label
            if len(metric_names) == 1:
                metric = metric_names[0]
                threshold = metric_thresholds[score][metric][0]
                unit = '%' if 'ratio' in metric else 'mm'
                if group_idx == 0:
                    group_labels[group_idx] = f'{METRIC_TO_TITLE[metric]} ≤{threshold:.2f} {unit} (n = {last_tp_subjects})'
                else:
                    group_labels[group_idx] = f'{METRIC_TO_TITLE[metric]} >{threshold:.2f} {unit} (n = {last_tp_subjects})'
            else:
                metric1, metric2 = metric_names[0], metric_names[1]
                threshold1 = metric_thresholds[score][metric1][0]
                threshold2 = metric_thresholds[score][metric2][0]
                unit1 = '%' if 'ratio' in metric1 else 'mm'
                unit2 = '%' if 'ratio' in metric2 else 'mm'
                if group_idx == 0:
                    group_labels[group_idx] = f'{METRIC_TO_TITLE[metric1]} ≤{threshold1:.2f} {unit1} (n = {last_tp_subjects})'
                elif group_idx == 1:
                    group_labels[group_idx] = f'{METRIC_TO_TITLE[metric1]} >{threshold1:.2f} {unit1} & {METRIC_TO_TITLE[metric2]} ≤{threshold2:.2f} {unit2} (n = {last_tp_subjects})'
                else:
                    group_labels[group_idx] = f'{METRIC_TO_TITLE[metric1]} >{threshold1:.2f} {unit1} & {METRIC_TO_TITLE[metric2]} >{threshold2:.2f} {unit2} (n = {last_tp_subjects})'

            # Plot individual trajectories (thin, semi-transparent)
            for participant_id in group_data['participant_id'].unique():
                participant_data = group_data[group_data['participant_id'] == participant_id].sort_values('days')
                if len(participant_data) >= 2:
                    ax.plot(participant_data['days'], participant_data['score_value'],
                           color=group_colors[group_idx], alpha=0.15, linewidth=0.8, zorder=1)

            # Plot model-based fitted line from the single NLME model
            if lme_result is not None:
                days_range = np.linspace(group_data['days'].min(), group_data['days'].max(), 100)
                log_days_range = np.log(days_range + 1)
                pred_df = pd.DataFrame({'log_days': log_days_range, 'group_idx': group_idx})
                # Ensure the categorical levels match the training data (important for patsy).
                pred_df['group_idx'] = pd.Categorical(
                    pred_df['group_idx'], categories=df_long['group_idx'].cat.categories
                )

                predictions = lme_result.predict(exog=pred_df)

                ax.plot(days_range, predictions,
                        color=group_colors[group_idx], linewidth=3, zorder=3,
                        label=group_labels[group_idx])

                # Optional: print per-group intercept/slope implied by the fixed effects.
                fe = lme_result.fe_params
                base_intercept = float(fe.get('Intercept', np.nan))
                base_slope = float(fe.get('log_days', np.nan))
                if group_idx == 0:
                    intercept = base_intercept
                    slope = base_slope
                else:
                    g_term = f"C(group_idx)[T.{group_idx}]"
                    # statsmodels may order interaction names as 'log_days:C(group_idx)[T.x]' or vice versa.
                    int_a = f"log_days:C(group_idx)[T.{group_idx}]"
                    int_b = f"C(group_idx)[T.{group_idx}]:log_days"
                    intercept = base_intercept + float(fe.get(g_term, 0.0))
                    slope = base_slope + float(fe.get(int_a, fe.get(int_b, 0.0)))

                print(f"\n{'='*60}")
                print(f"Non-Linear Mixed-Effects Model for {score.upper()}, Group {group_idx}: {group_labels[group_idx]}")
                print(f"{'='*60}")
                print(f"Fixed-effects implied intercept: {intercept:.3f}")
                print(f"Fixed-effects implied log(days+1) coefficient: {slope:.4f}")
                print(f"Number of subjects: {group_data['participant_id'].nunique()}")
                print(f"Number of observations: {len(group_data)}")
            else:
                # Fallback: plot mean trajectory if model fitting failed.
                mean_data = group_data.groupby('days')['score_value'].mean().reset_index()
                ax.plot(mean_data['days'], mean_data['score_value'],
                        color=group_colors[group_idx], linewidth=3, zorder=3,
                        label=group_labels[group_idx])

        # Customize plot
        ax.set_xlabel('Days from Baseline', fontsize=FONT_SIZE)
        ax.set_ylabel(f'{CLINICAL_SCORES_TO_AXES[score]}', fontsize=FONT_SIZE)
        ax.tick_params(axis='both', labelsize=FONT_SIZE)

        # Reorder legend
        legend_order = [0, 2, 1] if num_groups == 3 else [0, 1]
        handles, labels = ax.get_legend_handles_labels()
        ordered_handles = [handles[i] for i in legend_order if i < len(handles)]
        ordered_labels = [labels[i] for i in legend_order if i < len(labels)]
        ax.legend(ordered_handles, ordered_labels, loc='upper left',
                 fontsize=FONT_SIZE - 4, framealpha=0.9)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set ylim
        ax.set_ylim(SCORE_TO_YLIM[score][0], SCORE_TO_YLIM[score][1])
        # Set xlim to 200 days (~6 months)
        ax.set_xlim(-5, 200)

        plt.tight_layout()

        # Save figure
        num_subjects = df['participant_id'].nunique()
        if len(metric_names) == 1:
            figure_fname = os.path.join(output_dir,
                                       f'trajectory_plot_NLME_{score}_{metric_names[0]}_{num_subjects}subjects.png')
            print(f'\nNLME trajectory plot for {score} by {metric_names[0]} saved as {figure_fname}')
        else:
            figure_fname = os.path.join(output_dir,
                                       f'trajectory_plot_NLME_{score}_{metric_names[0]}_{metric_names[1]}_{num_subjects}subjects.png')
            print(f'\nNLME trajectory plot for {score} by {metric_names[0]} and {metric_names[1]} saved as {figure_fname}')

        figure_fnames.append(figure_fname)
        plt.savefig(figure_fname, dpi=300)
        plt.close()

    # Combine plots
    combine_plots(figure_fnames, 'nlme', output_dir)


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
                        group_labels[group_idx] = f'{METRIC_TO_TITLE[metric]} ≤{threshold:.2f} {unit} (n = {last_tp_count})'
                    else:
                        group_labels[group_idx] = f'{METRIC_TO_TITLE[metric]} >{threshold:.2f} {unit} (n = {last_tp_count})'
                else:
                    # Hierarchical case (two metrics: lesion width --> total tissue bridge)
                    metric1, metric2 = metrics[0], metrics[1]
                    threshold1, threshold2 = thresholds[0][0], thresholds[0][1]
                    unit1 = '%' if 'ratio' in metric1 else 'mm'
                    unit2 = '%' if 'ratio' in metric2 else 'mm'
                    last_tp_count = len(group_data[group_idx][last_time_point])

                    if group_idx == 0:
                        group_labels[group_idx] = f'{METRIC_TO_TITLE[metric1]} ≤{threshold1:.2f} {unit1} (n = {last_tp_count})'
                    elif group_idx == 1:
                        group_labels[group_idx] = f'{METRIC_TO_TITLE[metric1]} >{threshold1:.2f} {unit1} & {METRIC_TO_TITLE[metric2]} ≤{threshold2:.2f} {unit2} (n = {last_tp_count})'
                    else:
                        group_labels[group_idx] = f'{METRIC_TO_TITLE[metric1]} >{threshold1:.2f} {unit1} & {METRIC_TO_TITLE[metric2]} >{threshold2:.2f} {unit2} (n = {last_tp_count})'

                print(f'{score}, {metrics}, Time Point {tp} ({tp_month}m): Mean {score}={group_mean:.2f}, N={len(group_values)}')

                # Only show label in legend for the first actual time point that has data
                ax.errorbar(tp_month, group_mean, yerr=group_ci,
                            fmt='o', color=group_colors[group_idx], ecolor=group_colors[group_idx], alpha=1,
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
    ax.set_xlabel('', fontsize=FONT_SIZE)
    ax.set_ylabel(f'{CLINICAL_SCORES_TO_AXES[score]}', fontsize=FONT_SIZE)

    # Set x-ticks at the actual month points (0, 1, 3, 6)
    ax.set_xticks(time_points_months)
    ax.set_xticklabels(['Baseline', '1-month', '3-month', '6-month'], fontsize=FONT_SIZE)

    # Add minor ticks only at the labeled tick positions for better visualization
    ax.tick_params(axis='x', which='minor', bottom=True, length=4)
    # Place minor ticks at the same positions as the major ticks
    ax.set_xticks(time_points_months, minor=True)
    # Set x and y-tick font size
    ax.tick_params(axis='y', labelsize=FONT_SIZE)
    ax.tick_params(axis='x', labelsize=FONT_SIZE)

    # Custom legend handles: horizontal line + circle marker
    # Reorder legend to show green, blue, red
    legend_order = [0, 2, 1] if num_groups == 3 else [0, 1]  # green, blue, red for 3 groups; green, red for 2 groups
    legend_handles = [
        Line2D([0], [0], color=group_colors[i], marker='o', linestyle='-',
               linewidth=2.5, markersize=8)
        for i in legend_order
        if i < num_groups and group_labels[i]
    ]
    legend_labels = [group_labels[i] for i in legend_order if i < num_groups and group_labels[i]]
    ax.legend(legend_handles, legend_labels, loc='upper left',
              fontsize=FONT_SIZE - 4, framealpha=0.9)

    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set ylim to 110% of the maximum possible score to better fit the legend
    ax.set_ylim(SCORE_TO_YLIM[score][0], SCORE_TO_YLIM[score][1])
    # Set xlim to 200 days (~6 months)
    ax.set_xlim(-5, 200)



def main():

    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Create output directory
    output_dir = args.o
    os.makedirs(output_dir, exist_ok=True)

    # -----------
    # CSV file with lesion metrics
    # -----------
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

    # ----------------
    # Time points for clinical assessments
    # ----------------
    if args.file_timepoints:
        df_timepoints = pd.read_excel(args.file_timepoints, engine='openpyxl')
        # # If session_id is empty, fill with 'ses-01'
        # df_timepoints['session_id'] = df_timepoints['session_id'].fillna('ses-01')
        # Replace 'NT' with NaN
        df_timepoints = df_timepoints.replace('NT', np.nan)

        # Merge time points into main dataframe
        df = pd.merge(df, df_timepoints, on=['participant_id'], how='left')

    #----------------
    # Clinical scores and baseline metrics over time
    #----------------
    create_trajectory_plots(df, output_dir)


if __name__ == '__main__':
    main()
