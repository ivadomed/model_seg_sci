"""
Plot the lesion metrics obtained using different methods (manual vs automatic).

The script:
- reads CSV with lesion metrics computed using sct_analyze_lesion and aggregated across subjects
- reads XLSX file with manually measured lesion metrics (including clinical scores)
- merges the dataframes
- creates scatter plots with linear regression lines for each metric
- creates Bland-Altman Mean Difference plot for each metric
- creates plots of clinical scores over time for each participant with baseline metrics

Example usage:
    python 03_generate_figures_two_methods.py
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

from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


METRICS = ['midsagittal_length', 'midsagittal_width', 'ventral_tissue_bridge', 'dorsal_tissue_bridge', 'total_tissue_bridge']
METRIC_TO_TITLE = {
    'midsagittal_length': 'Midsagittal Lesion Length [mm]',
    'midsagittal_width': 'Midsagittal Lesion Width [mm]',
    'ventral_tissue_bridge': 'Midsagittal Ventral Tissue Bridges [mm]',
    'dorsal_tissue_bridge': 'Midsagittal Dorsal Tissue Bridges [mm]',
    'total_tissue_bridge': 'Midsagittal Total Tissue Bridges [mm]'
}

CLINICAL_SCORES_TO_AXES = {
    'uems': 'UEMS',
    'lems': 'LEMS',
    'ms': 'Motor Score',
    'pp': 'Pinprick Score',
    'lt': 'Light-Touch Score'
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
        help='Absolute path to an XLSX file with manually measured lesion metrics and clinical scores. '
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

    # Drop 'comment' column and unnamed columns
    df_manual = df_manual.drop(columns=['comment'], errors='ignore')
    unnamed_cols = [col for col in df_manual.columns if 'Unnamed' in col]
    df_manual = df_manual.drop(columns=unnamed_cols, errors='ignore')

    print(f'Read {len(df_manual)} rows from the manual metrics file: {file}')
    return df_manual


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


def compute_regression(x, y):
    """
    Compute a linear regression between x and y:
    y = Slope * x + Intercept
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

    You can then plot the linear fit by
    ax.plot(x_vals, y_vals, '--', color='red')

    :param x: ndarray: input - regressor
    :param y: ndarray: output - response
    :return: intercept: ndarray: intercept constant (bias term)
    :return: slope: ndarray: slope
    :return: reg_predictor: ndarray:
    :return: r2_sc: float: coefficient of determination
    :return x_vals: ndarray: x values for the linear fit plot
    :return y_vals: ndarray: y values for the linear fit plot
    """
    # Make sure we are working with numpy arrays
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()

    # Create an instance of the class LinearRegression, which will represent the regression model
    linear_regression = LinearRegression()
    # Perform linear regression (compute slope and intercept)
    linear_regression.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    intercept = linear_regression.intercept_        # underscore indicates that an attribute is estimated
    slope = linear_regression.coef_                 # underscore indicates that an attribute is estimated

    # Get x and y values to plot the linear fit
    x_vals = np.array([x.min(), x.max()])
    y_vals = intercept + slope * x_vals
    y_vals = np.squeeze(y_vals)                     # change shape from (1,N) to (N,)

    # Compute prediction (pass the regressor as the argument and get the corresponding predicted response)
    # Identical as reg_predictor = slope * x + intercept
    reg_predictor = linear_regression.predict(x.reshape(-1, 1))

    # Compute coefficient of determination R^2 of the prediction
    r2_sc = linear_regression.score(x.reshape(-1, 1), y.reshape(-1, 1))

    return intercept, slope, reg_predictor, r2_sc, x_vals, y_vals


def create_scatterplot(df, output_dir):
    """
    Create scatter plots with linear regression lines for each metric
    :param df: pandas dataframe with lesion metrics
    :param output_dir: output directory
    """

    # Set font to Arial
    plt.rcParams['font.sans-serif'] = 'Arial'

    for metric in METRICS:
        df_plot = df[[f'{metric}_manual', f'{metric}_sct']]
        # Drop rows with NaN values
        df_plot = df_plot.dropna()

        fig, axes = plt.subplots(figsize=(5, 5))

        max_val = df_plot.max().max()
        min_val = df_plot.min().min()

        ax = axes
        x = df_plot[f'{metric}_manual']
        y = df_plot[f'{metric}_sct']

        ax.scatter(x, y, s=90, alpha=0.5)
        ax.set_xlim(-0.1 * max_val, 1.1 * max_val)
        ax.set_ylim(-0.1 * max_val, 1.1 * max_val)

        # Add regression line
        intercept, slope, _, r2_sc, x_vals, y_vals = compute_regression(x, y)
        ax.plot(x_vals, y_vals, '--', color='red')

        # Compute Spearman correlation
        spearman_corr, p_value = stats.spearmanr(x, y, nan_policy='omit')
        ax.text(0.05, 0.95, f'Spearman ρ = {spearman_corr:.2f}\np = {p_value:.3g}',
                transform=ax.transAxes, verticalalignment='top', fontsize=FONT_SIZE, color='red')

        # Add diagonal line
        ax.plot([min_val, max_val], [min_val, max_val], ls='--', c='gray')

        # Change axes labels
        ax.set_title(f'{METRIC_TO_TITLE[metric]}', fontsize=FONT_SIZE)
        ax.set_xlabel(f'Manual', fontsize=FONT_SIZE)
        ax.set_ylabel(f'Automatic (from manual GTs)', fontsize=FONT_SIZE)

        if metric == 'midsagittal_length':
            # Change axes ticks to 0, 50, 100, 150, 200
            ax.set_xticks([0, 50, 100, 150, 200])
            ax.set_yticks([0, 50, 100, 150, 200])

        # Remove the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        # Save the plot
        figure_fname = os.path.join(output_dir, f'{metric}_manual_vs_sct_scatterplot.png')
        plt.savefig(figure_fname, dpi=300)
        print(f'Pairplot for {metric} saved as {figure_fname}')
        plt.close()


def create_scatterplot_3D_length_width(df, output_dir):
    """
    Create scatter plots with linear regression lines for each metric
        - between 3D length and manual midsagittal length
        - between 3D width and manual midsagittal width
    :param df: pandas dataframe with lesion metrics
    :param output_dir: output directory
    """

    # Set font to Arial
    plt.rcParams['font.sans-serif'] = 'Arial'

    for metric in ['length', 'width']:
        df_plot = df[[f'midsagittal_{metric}_manual', f'{metric}_sct']]

        fig, axes = plt.subplots(figsize=(5, 5))

        max_val = df_plot.max().max()
        min_val = df_plot.min().min()

        ax = axes
        x = df_plot[f'midsagittal_{metric}_manual']
        y = df_plot[f'{metric}_sct']

        ax.scatter(x, y, s=90, alpha=0.5)
        ax.set_xlim(-0.1 * max_val, 1.1 * max_val)
        ax.set_ylim(-0.1 * max_val, 1.1 * max_val)

        # Add regression line
        intercept, slope, _, r2_sc, x_vals, y_vals = compute_regression(x, y)
        ax.plot(x_vals, y_vals, '--', color='red')

        # Compute Spearman correlation
        spearman_corr, p_value = stats.spearmanr(x, y, nan_policy='omit')
        ax.text(0.05, 0.95, f'Spearman ρ = {spearman_corr:.2f}\np = {p_value:.3g}',
                transform=ax.transAxes, verticalalignment='top', fontsize=FONT_SIZE, color='red')

        # Add diagonal line
        ax.plot([min_val, max_val], [min_val, max_val], ls='--', c='gray')

        # Change axes labels
        ax.set_xlabel(f'Manual midsagittal {metric} [mm]', fontsize=FONT_SIZE)
        ax.set_ylabel(f'Automatic (from manual GTs) 3D {metric} [mm]', fontsize=FONT_SIZE)

        if metric == 'length':
            # Change axes ticks to 0, 50, 100, 150, 200
            ax.set_xticks([0, 50, 100, 150, 200])
            ax.set_yticks([0, 50, 100, 150, 200])

        # Remove the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        # Save the plot
        figure_fname = os.path.join(output_dir, f'{metric}_manual_sct3D_scatterplot.png')
        plt.savefig(figure_fname, dpi=200)
        print(f'Pairplot for 3D {metric} saved as {figure_fname}')
        plt.close()


def create_diff_plot(df, output_dir):
    """
    Create a Bland-Altman Mean Difference Plot for each metric
    https://www.statsmodels.org/devel/generated/statsmodels.graphics.agreement.mean_diff_plot.html
    :param df: pandas dataframe with lesion metrics
    :param output_dir: output directory
    """

    # Set font to Arial
    plt.rcParams['font.sans-serif'] = 'Arial'

    for metric in METRICS:
        df_plot = df[[f'{metric}_manual', f'{metric}_sct']]

        fig, axes = plt.subplots(figsize=(5, 5))

        ax = axes
        x = df_plot[f'{metric}_manual']
        y = df_plot[f'{metric}_sct']

        sm.graphics.mean_diff_plot(
            x, y,
            sd_limit=1.96,  # The default of 1.96 will produce 95% confidence intervals for the means of the differences
            ax=ax,
            scatter_kwds={
                's': 90,
                'alpha': 0.5,
            },
            mean_line_kwds={
                'color': 'black',
                'linestyle': '-',
                'alpha': 0.5,
                'linewidth': 2
            },
            limit_lines_kwds={
                'color': 'black',
                'linestyle': '--',
                'alpha': 0.5,
                'linewidth': 2
            }
        )

        # Set plot title and labels
        ax.set_title(f'{METRIC_TO_TITLE[metric]}\nManual vs Automatic (from manual GTs)', fontsize=FONT_SIZE)
        ax.set_xlabel(f'Mean', fontsize=FONT_SIZE)
        ax.set_ylabel(f'Difference', fontsize=FONT_SIZE)

        # Get the limits and means for custom styling
        diff = x - y            # Difference between x and y
        sd = np.std(diff)       # Standard deviation of the difference
        # Adjust y-lim
        ax.set_ylim(-1.96 * sd * 1.5, 1.96 * sd * 1.5)

        # Remove the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Draw dashed gray horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        plt.tight_layout()

        # Save the plot
        figure_fname = os.path.join(output_dir, f'{metric}_manual_vs_sct_diffplot.png')
        plt.savefig(figure_fname, dpi=300)
        print(f'Diffplot for {metric} saved as {figure_fname}')
        plt.close()


def create_clinical_metrics_plots(df_ses_01, output_dir):
    """
    Create an individual trajectory plot showing clinical scores across time points for each participant,
    with lines colored by baseline lesion metrics.

    :param df_ses_01: pandas dataframe with baseline lesion metrics (ses-01 sessions only) and clinical scores across
    multiple time points
    :param output_dir: output directory
    """
    # Set font to Arial
    plt.rcParams['font.sans-serif'] = 'Arial'

    clinical_scores = ('uems', 'lems', 'ms', 'pp', 'lt')
    # Sort time points in a logical order: bl, 1m, 3m, 6m, 12m
    time_point_order = {'bl': 0, '1m': 1, '3m': 2, '6m': 3, '12m': 4}
    time_points = sorted(list(time_point_order.keys()), key=lambda x: time_point_order.get(x, 99))

    # Convert any string clinical score columns to numeric
    for col in df_ses_01.columns:
        if col.startswith(clinical_scores):
            df_ses_01[col] = pd.to_numeric(df_ses_01[col], errors='coerce')

    # Loop over each clinical score
    for score in clinical_scores:
        # Filtering for UEMS
        if score == 'uems':
            # For UEMS, keep only subjects with 'tetrapara_bl' == 0
            #   0: tetraplegic
            #   1: paraplegic -- max UEMS at baseline (no impairment)
            df_ses_01_plot = df_ses_01[df_ses_01['tetrapara_bl'] == 0]
        else:
            df_ses_01_plot = df_ses_01

        # Loop over each lesion metric
        for metric in METRICS:
            # Create a figure for all participants
            fig, ax = plt.subplots(figsize=(10, 6))

            metric_name = f'{metric}_sct'   # automatic lesion metric from SCT

            # To stratify patients, determine median of the metric
            median_value = df_ses_01[metric_name].median()
            short_group_ids = []
            long_group_ids = []
            # Collect data for mean trajectories per group
            short_group_data = {tp: [] for tp in time_points}
            long_group_data = {tp: [] for tp in time_points}

            # For colormap
            min_value = df_ses_01[metric_name].min()
            max_value = df_ses_01[metric_name].max()

            # Process each participant
            for participant_id in df_ses_01['participant_id'].unique():
                # Get participant data
                participant_data = df_ses_01[df_ses_01['participant_id'] == participant_id]
                # Check if participant has clinical data and the metric
                if (participant_id not in df_ses_01_plot['participant_id'].values or
                    metric_name not in participant_data.columns or
                    pd.isna(participant_data[metric_name].values[0])):
                    continue

                participant_clinical = df_ses_01_plot[df_ses_01_plot['participant_id'] == participant_id]

                # Get metric value for this participant (for coloring)
                metric_value = participant_data[metric_name].values[0]
                # Determine if this participant belongs to short or long group
                if metric_value <= median_value:
                    group = 'short'
                    short_group_ids.append(participant_id)
                else:
                    group = 'long'
                    long_group_ids.append(participant_id)

                # Get clinical score values across time points
                time_points_present = [tp for tp in time_points if f'{score}_{tp}' in participant_clinical.columns]

                if not time_points_present:
                    continue

                # Get values for this score across time points
                time_values = []
                score_values = []

                for tp_idx, tp in enumerate(time_points_present):
                    col_name = f'{score}_{tp}'
                    if col_name in participant_clinical.columns and not pd.isna(participant_clinical[col_name].values[0]):
                        val = float(participant_clinical[col_name].values[0])  # Ensure value is float
                        time_values.append(tp_idx)  # Use index for x-axis to make equal spacing
                        score_values.append(val)

                        # Add to group data for mean trajectory
                        if group == 'short':
                            short_group_data[tp].append(val)
                        else:
                            long_group_data[tp].append(val)

                if len(time_values) < 2:  # Need at least 2 points to draw a line
                    continue

                # Normalize metric value for colormap (0-1 range)
                norm_value = (metric_value - min_value) / (max_value - min_value) if max_value > min_value else 0.5

                # Calculate marker size based on metric value (scale between 2-12)
                marker_size = 2 + 10 * norm_value  # Scale normalized value to range between 2-12

                # Baseline markers modulated by metric the baseline lesion metric value
                ax.plot(time_values[0], score_values[0], 'o-', alpha=0.5,
                        color=plt.cm.cool(norm_value),
                        markersize=marker_size)
                # Trajectory lines
                ax.plot(time_values, score_values, 'o-', alpha=0.5,
                        color=plt.cm.cool(norm_value),
                        linewidth=1, markersize=1)

            # Calculate and plot mean ± standard error (SE) trajectories for each group
            for tp_idx, tp in enumerate(time_points):
                # Short group
                short_values = short_group_data[tp]
                if short_values:
                    # Ensure all values are numeric
                    short_values = [float(val) for val in short_values]
                    short_mean = np.mean(short_values)
                    short_se = np.std(short_values) / np.sqrt(len(short_values)) if len(short_values) > 1 else 0
                    # Include sample size in the legend label
                    label = f'Short {METRIC_TO_TITLE[metric].split("[")[0]} (≤{median_value:.1f} mm, n={len(short_group_ids)})'
                    ax.errorbar(tp_idx, short_mean, yerr=short_se,
                                fmt='o', color='blue', ecolor='blue',
                                markersize=5, capsize=5,
                                label=label)

                # Long group
                long_values = long_group_data[tp]
                if long_values:
                    # Ensure all values are numeric
                    long_values = [float(val) for val in long_values]
                    long_mean = np.mean(long_values)
                    long_se = np.std(long_values) / np.sqrt(len(long_values)) if len(long_values) > 1 else 0
                    # Include sample size in the legend label
                    label = f'Long {METRIC_TO_TITLE[metric].split("[")[0]} (>{median_value:.1f} mm, n={len(long_group_ids)})' if tp_idx == 1 else ""
                    ax.errorbar(tp_idx, long_mean, yerr=long_se,
                                fmt='o', color='red', ecolor='red',
                                markersize=5, capsize=5,
                                label=label)

            # Connect mean points with lines
            mean_short_x = []
            mean_short_y = []
            mean_long_x = []
            mean_long_y = []

            for tp_idx, tp in enumerate(time_points):
                short_values = short_group_data[tp]
                if short_values:
                    # Ensure all values are numeric
                    short_values = [float(val) for val in short_values]
                    mean_short_x.append(tp_idx)
                    mean_short_y.append(np.mean(short_values))

                long_values = long_group_data[tp]
                if long_values:
                    # Ensure all values are numeric
                    long_values = [float(val) for val in long_values]
                    mean_long_x.append(tp_idx)
                    mean_long_y.append(np.mean(long_values))

            if len(mean_short_x) > 1:
                ax.plot(mean_short_x, mean_short_y, '-', color='blue', linewidth=2.5)

            if len(mean_long_x) > 1:
                ax.plot(mean_long_x, mean_long_y, '-', color='red', linewidth=2.5)

            # Set labels and title
            ax.set_title(f'{CLINICAL_SCORES_TO_AXES[score]} over time stratified by '
                         f'{METRIC_TO_TITLE[metric].split("[")[0]}', fontsize=FONT_SIZE+2)
            ax.set_xlabel('Time Point', fontsize=FONT_SIZE)
            ax.set_ylabel(f'{CLINICAL_SCORES_TO_AXES[score]}', fontsize=FONT_SIZE)
            ax.set_xticks(range(len(time_points)))
            ax.set_xticklabels(['Baseline', 'M1', 'M3', 'M6', 'M12'], fontsize=FONT_SIZE)

            # Add color bar for lesion metric
            sm = plt.cm.ScalarMappable(cmap=plt.cm.cool,
                                       norm=plt.Normalize(vmin=min_value, vmax=max_value))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label(METRIC_TO_TITLE[metric], fontsize=FONT_SIZE)

            # Add legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=FONT_SIZE-2, framealpha=0.9)

            # Remove the top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Save the plot
            plt.tight_layout()
            figure_fname = os.path.join(output_dir, f'{score}_by_{metric}_trajectory_plot.png')
            plt.savefig(figure_fname, dpi=300)
            print(f'Trajectory plot for {score} by {metric} saved as {figure_fname}')
            plt.close()


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
    df_manual = read_file_manual(file_manual)

    #----------------
    # Merge the dataframes
    #----------------
    df = pd.merge(df_sct, df_manual, on=['participant_id', 'session_id'])

    #----------------
    # Create new dataframes for ses-01 and multiple sessions
    #----------------
    # Keep only subjects with more than 1 session
    df_multiple_ses = df.groupby('participant_id').filter(lambda x: len(x) > 1 and x['session_id'].nunique() > 1)
    # Keep only 'ses-01' sessions
    df_ses_01 = df[df['session_id'] == 'ses-01']
    # Print number of subjects
    print(f'ses-01: Number of subjects: {df_ses_01.shape[0]}')
    print(f'Multiple sessions: Number of subjects: {df_multiple_ses.shape[0]}')

    #----------------
    # Filter subjects based on MRI time since injury
    #----------------
    # Convert mri_time_since_injury to numeric (in days)
    df_ses_01['mri_time_since_injury'] = pd.to_numeric(df_ses_01['mri_time_since_injury'], errors='coerce')
    desc = df_ses_01['mri_time_since_injury'].describe()
    print(f'Description of MRI Time Since Injury (in days):\n{desc}')
    # Keep only subjects with mri_time_since_injury (in days) from 12 days to 2 months
    df_ses_01 = df_ses_01[(df_ses_01['mri_time_since_injury'] >= 12) & (df_ses_01['mri_time_since_injury'] <= 60)]
    desc = df_ses_01['mri_time_since_injury'].describe()
    print(f'Description of MRI Time Since Injury (in days):\n{desc}')

    # # Keep only test subjects (i.e., those who were not used for SCIsegV2 training)
    # # https://github.com/ivadomed/model_seg_sci/blob/main/dataset-conversion/dataset_split_seed710.yaml
    # # Note: The following subjects were obtained using Claude
    # test_subjects = ['sub-zh03_ses-01', 'sub-zh06_ses-01', 'sub-zh06_ses-02', 'sub-zh15_ses-01', 'sub-zh18_ses-01',
    #                  'sub-zh19_ses-01', 'sub-zh22_ses-01', 'sub-zh22_ses-02', 'sub-zh27_ses-01', 'sub-zh30_ses-01',
    #                  'sub-zh34_ses-01', 'sub-zh54_ses-01', 'sub-zh66_ses-01']
    # # Create a new column that combines participant_id and session_id
    # df['combined_id'] = df['participant_id'] + '_' + df['session_id']
    # # Filter the dataframe to keep only the test subjects
    # df_filtered = df[df['combined_id'].isin(test_subjects)]
    # # If you want to remove the 'combined_id' column after filtering:
    # df = df_filtered.drop('combined_id', axis=1)

    # # Exclude sub-zh15, sub-zh81
    # df = df[~df['participant_id'].isin(['sub-zh15', 'sub-zh81'])]
    # print(len(df))

    #----------------
    # Plotting
    #----------------
    output_dir = args.o
    # create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Scatter plot with linear regression lines
    create_scatterplot(df_ses_01, output_dir)
    # Scatter plot for 3D lesion length and width
    create_scatterplot_3D_length_width(df, output_dir)
    # Bland-Altman Mean Difference Plot
    create_diff_plot(df_ses_01, output_dir)

    #----------------
    # Clinical scores and baseline metrics over time
    #----------------
    create_clinical_metrics_plots(df_ses_01, output_dir)


if __name__ == '__main__':
    main()
