"""
Generate correlation matrices between three methods for each lesion metric.

The script:
- reads XLSX file with manually measured lesion metrics
- reads CSV files with lesion metrics computed using sct_analyze_lesion for both GT and SCIsegV2 methods
- merges the dataframes
- creates correlation matrices for each metric showing correlations between manual, semi-automatic (GT), and automatic (SCIsegV2) methods
- computes both Pearson and Spearman correlations with statistical significance
- creates scatter plots with linear regression lines for each metric
- creates Bland-Altman Mean Difference plot for each metric

Example usage:
    python 03_generate_lesion_metric_plots.py
        -file-gt <PATH_TO_GT_CSV_FILE>
        -file-scisegv2 <PATH_TO_SCISEGV2_CSV_FILE>
        -file-manual <PATH_TO_XLSX_FILE>
        -o <OUTPUT_DIR>

Note: to read XLS files, you might need to install the following packages:
    pip install openpyxl seaborn

Author: Jan Valosek
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import subprocess
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


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

FONT_SIZE = 12


def get_parser():
    """
    parser function
    """
    parser = argparse.ArgumentParser(
        description='Generate correlation matrices between manual, semi-automatic (GT), and automatic (SCIsegV2) methods for each lesion metric.',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-file-gt',
        required=True,
        type=str,
        help='Absolute path to a CSV file with lesion metrics computed using sct_analyze_lesion on GT (manual) lesion masks.'
    )
    parser.add_argument(
        '-file-scisegv2',
        required=True,
        type=str,
        help='Absolute path to a CSV file with lesion metrics computed using sct_analyze_lesion on SCIsegV2 (automatic) lesion masks.'
    )
    parser.add_argument(
        '-file-manual',
        required=True,
        type=str,
        help='Absolute path to an XLSX file with manually measured lesion metrics.'
    )
    parser.add_argument(
        '-o',
        required=True,
        type=str,
        help='Path to the output folder where correlation matrices will be saved.'
    )

    return parser


def read_file_manual(file):
    """
    Read the XLSX file with manually measured metrics.
    :param file: str: path to the XLSX file
    :return df_manual: pandas DataFrame: dataframe with manually measured lesion metrics
    """
    df_manual = pd.read_excel(file)
    # Drop rows where 'participant_id' is NaN or 'exclude' -- rows with comments
    df_manual = df_manual.dropna(subset=['participant_id'])
    df_manual = df_manual[df_manual['participant_id'] != 'exclude']
    # If session_id is nan in the manual file, set it to 'ses-01'
    df_manual['session_id'] = df_manual['session_id'].fillna('ses-01')

    # Sum up ventral and dorsal tissue bridges to get total tissue bridge
    df_manual['total_tissue_bridge'] = df_manual['ventral_tissue_bridge'] + df_manual['dorsal_tissue_bridge']

    # Compute tissue bridge ratios
    df_manual['dorsal_bridge_ratio'] = df_manual.apply(
        lambda row: (row['dorsal_tissue_bridge'] / row['total_tissue_bridge'] * 100)
        if row['total_tissue_bridge'] > 0 else 0, axis=1)
    df_manual['ventral_bridge_ratio'] = df_manual.apply(
        lambda row: (row['ventral_tissue_bridge'] / row['total_tissue_bridge'] * 100)
        if row['total_tissue_bridge'] > 0 else 0, axis=1)

    # Drop 'comment' column and unnamed columns
    df_manual = df_manual.drop(columns=['comment'], errors='ignore')
    unnamed_cols = [col for col in df_manual.columns if 'Unnamed' in col]
    df_manual = df_manual.drop(columns=unnamed_cols, errors='ignore')

    # Add suffix to distinguish from other methods
    metric_cols = list(METRIC_TO_TITLE.keys())
    for col in metric_cols:
        if col in df_manual.columns:
            df_manual.rename(columns={col: f'{col}_manual'}, inplace=True)

    print(f'Read {len(df_manual)} rows from the manual metrics file: {file}')
    print(f'Number of unique participants in the manual metrics file: {df_manual['participant_id'].nunique()}')
    return df_manual


def read_file_sct(file_sct, method_suffix):
    """
    Read CSV file with lesion metrics computed using sct_analyze_lesion.
    :param file_sct: str: path to the CSV file
    :param method_suffix: str: suffix to add to metric columns (e.g., 'gt' or 'scisegv2')
    :return df_sct: pandas DataFrame: dataframe with SCT-computed lesion metrics
    """
    df_sct = pd.read_csv(file_sct)

    # Drop 'number_of_lesions', 'volume', 'length', and 'width' columns
    df_sct = df_sct.drop(columns=['number_of_lesions', 'volume', 'length', 'width'], errors='ignore')

    # Rename columns to match the manual metrics
    df_sct.rename(columns={'length_interpolated_midsagittal_slice': 'midsagittal_length',
                           'width_interpolated_midsagittal_slice': 'midsagittal_width',
                           'interpolated_dorsal_bridge_width': 'dorsal_tissue_bridge',
                           'interpolated_ventral_bridge_width': 'ventral_tissue_bridge',
                           'interpolated_total_bridge_width': 'total_tissue_bridge'},
                  inplace=True)

    # Compute tissue bridge ratios
    df_sct['dorsal_bridge_ratio'] = df_sct.apply(
        lambda row: (row['dorsal_tissue_bridge'] / row['total_tissue_bridge'] * 100)
        if row['total_tissue_bridge'] > 0 else 0, axis=1)
    df_sct['ventral_bridge_ratio'] = df_sct.apply(
        lambda row: (row['ventral_tissue_bridge'] / row['total_tissue_bridge'] * 100)
        if row['total_tissue_bridge'] > 0 else 0, axis=1)

    # Add suffix to metric columns except participant_id and session_id
    metric_cols = list(METRIC_TO_TITLE.keys())
    for col in metric_cols:
        if col in df_sct.columns:
            df_sct.rename(columns={col: f'{col}_{method_suffix}'}, inplace=True)

    print(f'Read {len(df_sct)} rows from the {method_suffix.upper()} metrics file: {file_sct}')
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


def create_correlation_matrix(df, metric, output_dir):
    """
    Create correlation matrix for a specific metric across the three methods.
    :param df: pandas DataFrame with all three methods' data
    :param metric: str: metric name (e.g., 'midsagittal_length')
    :param output_dir: str: output directory
    """
    # Set font to Arial
    plt.rcParams['font.sans-serif'] = 'Arial'

    # Define method columns for this metric
    method_cols = [f'{metric}_manual', f'{metric}_gt', f'{metric}_scisegv2']
    method_labels_x = ['Manual', 'Semi-automatic\n(GT + SCT)']
    method_labels_y = ['Semi-automatic\n(GT + SCT)', 'Automatic\n(SCIsegV2 + SCT)']

    # Extract data for this metric, dropping rows with any NaN values
    df_metric = df[['participant_id', 'session_id'] + method_cols].dropna()

    if len(df_metric) == 0:
        print(f'No valid data for {metric}, skipping...')
        return

    # Create correlation data
    corr_data = df_metric[method_cols]

    # Compute Pearson and Spearman correlations
    pearson_corr = corr_data.corr(method='pearson')
    spearman_corr = corr_data.corr(method='spearman')

    # Compute p-values for correlations
    n_methods = len(method_cols)
    pearson_pvals = np.full((n_methods, n_methods), np.nan)
    spearman_pvals = np.full((n_methods, n_methods), np.nan)

    for i in range(n_methods):
        for j in range(n_methods):
            if i != j:
                # Pearson p-value
                _, p_pearson = stats.pearsonr(corr_data.iloc[:, i], corr_data.iloc[:, j])
                pearson_pvals[i, j] = p_pearson

                # Spearman p-value
                _, p_spearman = stats.spearmanr(corr_data.iloc[:, i], corr_data.iloc[:, j])
                spearman_pvals[i, j] = p_spearman

    # Remove the first row and the last column from pearson_corr to reduce it from 3x3 to 2x2
    pearson_corr = pearson_corr.iloc[1:, :-1]
    spearman_corr = spearman_corr.iloc[1:, :-1]
    pearson_pvals = pearson_pvals[1:, :-1]
    spearman_pvals = spearman_pvals[1:, :-1]

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pearson correlation matrix
    mask_pearson = np.array([[False, True], [False, False]], dtype=bool)    # 2x2
    sns.heatmap(pearson_corr, mask=mask_pearson, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                xticklabels=method_labels_x, yticklabels=method_labels_y,
                vmin=-1, vmax=1, fmt='.3f', ax=ax1, annot_kws={'size': FONT_SIZE})
    ax1.set_title(f'Pearson Correlation\n{METRIC_TO_TITLE[metric]}', fontsize=FONT_SIZE + 2)

    # Add p-values as text annotations
    for i in range(n_methods-1):    # 3x3 --> 2x2
        for j in range(n_methods-1):    # 3x3 --> 2x2
            p_val = pearson_pvals[i, j]
            if not np.isnan(p_val):
                ax1.text(j + 0.5, i + 0.75, format_pvalue(p_val),
                        ha='center', va='center', fontsize=FONT_SIZE - 2, color='black')

    # Spearman correlation matrix
    mask_spearman = np.array([[False, True], [False, False]], dtype=bool)
    sns.heatmap(spearman_corr, mask=mask_spearman, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                xticklabels=method_labels_x, yticklabels=method_labels_y,
                vmin=-1, vmax=1, fmt='.3f', ax=ax2, annot_kws={'size': FONT_SIZE})
    ax2.set_title(f'Spearman Correlation\n{METRIC_TO_TITLE[metric]}', fontsize=FONT_SIZE + 2)

    # Add p-values as text annotations
    for i in range(n_methods-1):  # 3x3 --> 2x2
        for j in range(n_methods-1):  # 3x3 --> 2x2
            p_val = spearman_pvals[i, j]
            if not np.isnan(p_val):
                ax2.text(j + 0.5, i + 0.75, format_pvalue(p_val),
                        ha='center', va='center', fontsize=FONT_SIZE - 2, color='black')

    plt.tight_layout()

    # Save the plot
    num_subjects = len(df_metric)
    figure_fname = os.path.join(output_dir, f'correlation_matrix_{metric}_{num_subjects}subjects.png')
    plt.savefig(figure_fname, dpi=300, bbox_inches='tight')
    print(f'Correlation matrix for {metric} saved as {figure_fname}')
    plt.close()

    # # Print correlation summary to console
    # print(f'\n--- {METRIC_TO_TITLE[metric]} ({num_subjects} subjects) ---')
    # print('Pearson correlations:')
    # for i in range(n_methods):
    #     for j in range(i + 1, n_methods):
    #         corr_val = pearson_corr.iloc[i, j]
    #         p_val = pearson_pvals[i, j]
    #         print(f'  {method_labels[i]} vs {method_labels[j]}: r = {corr_val:.3f}, {format_pvalue(p_val)}')
    #
    # print('Spearman correlations:')
    # for i in range(n_methods):
    #     for j in range(i + 1, n_methods):
    #         corr_val = spearman_corr.iloc[i, j]
    #         p_val = spearman_pvals[i, j]
    #         print(f'  {method_labels[i]} vs {method_labels[j]}: ρ = {corr_val:.3f}, {format_pvalue(p_val)}')


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
    # First row: midsagittal_length and midsagittal_width
    cmd_row1 = f"convert {os.path.join(output_dir, f'{figure_type}_midsagittal_length_{num_subjects}subjects.png')} " \
               f"{os.path.join(output_dir, f'{figure_type}_midsagittal_width_{num_subjects}subjects.png')} " \
               f"+append {os.path.join(output_dir, f'temp1_{num_subjects}subjects.png')}"
    # Second row: ventral_tissue_bridge, dorsal_tissue_bridge, and total_tissue_bridge
    cmd_row2 = f"convert {os.path.join(output_dir, f'{figure_type}_ventral_tissue_bridge_{num_subjects}subjects.png')} " \
               f"{os.path.join(output_dir, f'{figure_type}_dorsal_tissue_bridge_{num_subjects}subjects.png')} " \
               f"{os.path.join(output_dir, f'{figure_type}_total_tissue_bridge_{num_subjects}subjects.png')} " \
               f"+append {os.path.join(output_dir, f'temp2_{num_subjects}subjects.png')}"
    # Third row: ventral_bridge_ratio and dorsal_bridge_ratio
    cmd_row3 = f"convert {os.path.join(output_dir, f'{figure_type}_ventral_bridge_ratio_{num_subjects}subjects.png')} " \
               f"{os.path.join(output_dir, f'{figure_type}_dorsal_bridge_ratio_{num_subjects}subjects.png')} " \
               f"+append {os.path.join(output_dir, f'temp3_{num_subjects}subjects.png')}"
    # Combine all rows
    cmd_combine = f"convert {os.path.join(output_dir, f'temp1_{num_subjects}subjects.png')} " \
                  f"{os.path.join(output_dir, f'temp2_{num_subjects}subjects.png')} " \
                  f"{os.path.join(output_dir, f'temp3_{num_subjects}subjects.png')} " \
                  f"-append {os.path.join(combined_dir, f'{figure_type}_combined_{num_subjects}subjects.png')}; " \
                  f"rm {os.path.join(output_dir, f'temp1_{num_subjects}subjects.png')} " \
                  f"{os.path.join(output_dir, f'temp2_{num_subjects}subjects.png')} " \
                  f"{os.path.join(output_dir, f'temp3_{num_subjects}subjects.png')}"
    # Execute the commands
    subprocess.run(cmd_row1, shell=True)
    subprocess.run(cmd_row2, shell=True)
    subprocess.run(cmd_row3, shell=True)
    subprocess.run(cmd_combine, shell=True)
    print(
        f"Combined {figure_type} saved as {os.path.join(combined_dir, f'{figure_type}_combined_{num_subjects}subjects.png')}")


def create_scatterplot(df, output_dir, method):
    """
    Create scatter plots with linear regression lines for each metric
    :param df: pandas dataframe with lesion metrics
    :param output_dir: output directory
    :param method: str: method ('GT' or 'SCIsegV2')
    """

    # Set font to Arial
    plt.rcParams['font.sans-serif'] = 'Arial'

    for metric in METRIC_TO_TITLE.keys():
        df_plot = df[[f'{metric}_manual', f'{metric}_{method.lower()}']]
        # # Drop rows with NaN values
        # df_plot = df_plot.dropna()

        fig, axes = plt.subplots(figsize=(5, 5))

        max_val = df_plot.max().max()
        min_val = df_plot.min().min()

        ax = axes
        x = df_plot[f'{metric}_manual']
        y = df_plot[f'{metric}_{method.lower()}']

        ax.scatter(x, y, s=20, alpha=1, color='black', edgecolor='black')
        ax.set_xlim(-0.1 * max_val, 1.1 * max_val)
        ax.set_ylim(-0.1 * max_val, 1.1 * max_val)

        # Add regression line
        intercept, slope, _, r2_sc, x_vals, y_vals = compute_regression(x, y)
        ax.plot(x_vals, y_vals, '-', color='red')

        # Compute Spearman correlation
        spearman_corr, p_value = stats.spearmanr(x, y, nan_policy='omit')
        ax.text(0.05, 0.95, f'Spearman\nρ = {spearman_corr:.2f}\n{format_pvalue(p_value)}',
                transform=ax.transAxes, verticalalignment='top', fontsize=FONT_SIZE, color='black')

        # Add diagonal line
        ax.plot([min_val, max_val], [min_val, max_val], ls='--', c='gray')

        # Change axes labels
        ax.set_title(f'{METRIC_TO_TITLE[metric].split("[")[0]}', fontsize=FONT_SIZE)
        ax.set_xlabel(f'Manual', fontsize=FONT_SIZE)
        ax.set_ylabel(f'{METHOD_TO_TITLE[method]}', fontsize=FONT_SIZE)

        if metric == 'midsagittal_length':
            # Tweak axes ticks
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_yticks([0, 25, 50, 75, 100])

        # Remove the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        # Save the plot
        num_subjects = len(df_plot)
        figure_fname = os.path.join(output_dir, f'{METHOD_TO_FNAME[method]}_scatterplot_{metric}_{num_subjects}subjects.png')
        plt.savefig(figure_fname, dpi=300)
        print(f'Pairplot for {metric} saved as {figure_fname}')
        plt.close()

    combine_plot(f'{METHOD_TO_FNAME[method]}_scatterplot', num_subjects, output_dir)


def create_scatterplot_3D_length_width(df, output_dir, method):
    """
    Create scatter plots with linear regression lines for each metric
        - between 3D length and manual midsagittal length
        - between 3D width and manual midsagittal width
    :param df: pandas dataframe with lesion metrics
    :param output_dir: output directory
    :param method: str: method ('GT' or 'SCIsegV2')
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

        ax.scatter(x, y, s=20, alpha=1, color='black', edgecolor='black')
        ax.set_xlim(-0.1 * max_val, 1.1 * max_val)
        ax.set_ylim(-0.1 * max_val, 1.1 * max_val)

        # Add regression line
        intercept, slope, _, r2_sc, x_vals, y_vals = compute_regression(x, y)
        ax.plot(x_vals, y_vals, '-', color='red')

        # Compute Spearman correlation
        spearman_corr, p_value = stats.spearmanr(x, y, nan_policy='omit')
        ax.text(0.05, 0.95, f'Spearman\nρ = {spearman_corr:.2f}\n{format_pvalue(p_value)}',
                transform=ax.transAxes, verticalalignment='top', fontsize=FONT_SIZE, color='black')

        # Add diagonal line
        ax.plot([min_val, max_val], [min_val, max_val], ls='--', c='gray')

        # Change axes labels
        ax.set_xlabel(f'Manual midsagittal {metric} [mm]', fontsize=FONT_SIZE)
        ax.set_ylabel(f'{METHOD_TO_TITLE[method]} 3D {metric} [mm]', fontsize=FONT_SIZE)

        if metric == 'length':
            # Tweak axes ticks
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_yticks([0, 25, 50, 75, 100])

        # Remove the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        # Save the plot
        figure_fname = os.path.join(output_dir, f'{method}_{metric}_manual_sct3D_scatterplot_{len(df_plot)}subjects.png')
        plt.savefig(figure_fname, dpi=200)
        print(f'Pairplot for 3D {metric} saved as {figure_fname}')
        plt.close()


def create_diff_plot(df, output_dir, method):
    """
    Create a Bland-Altman Mean Difference Plot for each metric
    https://www.statsmodels.org/devel/generated/statsmodels.graphics.agreement.mean_diff_plot.html
    :param df: pandas dataframe with lesion metrics
    :param output_dir: output directory
    :method: str: method ('GT' or 'SCIsegV2')
    """

    # Set font to Arial
    plt.rcParams['font.sans-serif'] = 'Arial'

    for metric in METRIC_TO_TITLE.keys():
        df_plot = df[[f'{metric}_manual', f'{metric}_{method.lower()}']]

        fig, axes = plt.subplots(figsize=(5, 5))

        ax = axes
        x = df_plot[f'{metric}_manual']
        y = df_plot[f'{metric}_{method.lower()}']

        sm.graphics.mean_diff_plot(
            x, y,
            sd_limit=1.96,  # The default of 1.96 will produce 95% confidence intervals for the means of the differences
            ax=ax,
            scatter_kwds={
                's': 20,
                'alpha': 1,
                'color': 'black',
                'edgecolor': 'black',

            },
            mean_line_kwds={
                'color': 'black',
                'linestyle': '-',
                'alpha': 0.5,
                'linewidth': 1
            },
            limit_lines_kwds={
                'color': 'black',
                'linestyle': '--',
                'alpha': 0.5,
                'linewidth': 1
            }
        )

        # Set plot title and labels
        ax.set_title(f'{METRIC_TO_TITLE[metric].split("[")[0]}\n'
                     f'Manual vs {METHOD_TO_TITLE[method]}', fontsize=FONT_SIZE)
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
        num_subjects = len(df_plot)
        figure_fname = os.path.join(output_dir, f'{METHOD_TO_FNAME[method]}_diffplot_{metric}_{num_subjects}subjects.png')
        plt.savefig(figure_fname, dpi=300)
        print(f'Diffplot for {metric} saved as {figure_fname}')
        plt.close()

    combine_plot(f'{METHOD_TO_FNAME[method]}_diffplot', num_subjects, output_dir)


def main():
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Read the data files
    print('Reading data files...')
    # Manual
    df_manual = read_file_manual(args.file_manual)
    # Semi-automatic (GT)
    df_gt = read_file_sct(args.file_gt, 'gt')
    # Automatic (SCIsegV2)
    df_scisegv2 = read_file_sct(args.file_scisegv2, 'scisegv2')

    # Merge the dataframes
    print('Merging dataframes...')
    df = pd.merge(df_manual, df_gt, on=['participant_id', 'session_id'], how='inner')
    df = pd.merge(df, df_scisegv2, on=['participant_id', 'session_id'], how='inner')
    print(f'Total number of subjects after merging: {len(df)}')

    # Drop subjects with NaN values in any of the lesion metrics
    df = df.dropna(subset=[f'{metric}_manual' for metric in METRIC_TO_TITLE.keys()] +
                   [f'{metric}_gt' for metric in METRIC_TO_TITLE.keys()] +
                   [f'{metric}_scisegv2' for metric in METRIC_TO_TITLE.keys()])
    print(f'Number of subjects after dropping NaN values: {len(df)}')

    # Create output directory
    os.makedirs(args.o, exist_ok=True)

    # Generate correlation matrices for each metric
    print('\nGenerating correlation matrices...')
    for metric in METRIC_TO_TITLE.keys():
        create_correlation_matrix(df, metric, args.o)

    # ----------------
    # Create scatter plots and diff plots for manual vs semi-automatic (GT)
    # ----------------
    print('\nGenerating plots for Manual vs Semi-automatic (GT)...')
    create_scatterplot(df, args.o, 'GT')
    # create_scatterplot_3D_length_width(df, args.o, 'GT')
    create_diff_plot(df, args.o, 'GT')

    # ----------------
    # Create scatter plots and diff plots for manual vs automatic (SCIsegV2)
    # ----------------
    print('\nGenerating plots for Manual vs Automatic (SCIsegV2)...')
    create_scatterplot(df, args.o, 'SCIsegV2')
    # create_scatterplot_3D_length_width(df, args.o, 'SCIsegV2')
    create_diff_plot(df, args.o, 'SCIsegV2')

    print(f'\nAll plots and correlation matrices saved to: {args.o}')

if __name__ == '__main__':
    main()
