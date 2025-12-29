"""
Figure 3.

Generate scatter plots and Bland-Altman plots between manual and automatic lesion metrics.
The script generates a single figure with 3 scatter plots (first row) and 3 Bland-Altman plots (second row).

The script:
- reads CSV file with lesion metrics with _sct and _manual suffixes.
- for each lesion metric (lesion length, width, tissue bridges):
    - computes Spearman correlation between manual and automatic lesion metrics
    - creates scatter plot with linear regression line
    - creates Bland-Altman Mean Difference plot

Example usage:
    python 03_generate_lesion_metric_plots.py
        -i <PATH_TO_CSV_FILE>
        -o <OUTPUT_DIR>

Note: to read XLS files, you might need to install the following packages:
    pip install openpyxl seaborn

Author: Jan Valosek
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import subprocess
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from utils import read_csv_file_with_lesion_metrics

METRICS_AXIS_LIM_SCATTER = {
    'midsagittal_length': (-5, 200),
    'midsagittal_width': (-0.5, 11),
    'total_tissue_bridge': (-0.5, 8)
}

METRICS_AXIS_XLIM_DIFF = {
    'midsagittal_length': (-3, 75),
    'midsagittal_width': (-0.5, 11),
    'total_tissue_bridge': (-0.5, 6)
}

METRICS_AXIS_YLIM_DIFF = {
    'midsagittal_length': (-46, 46),
    'midsagittal_width': (-4.5, 4.5),
    'total_tissue_bridge': (-4.5, 4.5)
}

METRIC_TO_TITLE = {
    'midsagittal_length': 'Midsagittal Lesion Length [mm]',
    'midsagittal_width': 'Midsagittal Lesion Width [mm]',
    'total_tissue_bridge': 'Midsagittal Total Tissue Bridges [mm]'
}

FONT_SIZE = 19


def get_parser():
    """
    parser function
    """
    parser = argparse.ArgumentParser(
        description='Generate scatter plots and Bland-Altman plots between manual and automatic lesion metrics.',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-i',
        required=True,
        type=str,
        help='Absolute path to a CSV file with lesion metrics with _sct and _manual suffixes.'
    )
    parser.add_argument(
        '-o',
        required=True,
        type=str,
        help='Path to the output folder where correlation matrices will be saved.'
    )

    return parser


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


def combine_plot(figure_type, figure_fname_list, output_dir, panel_label=None):
    """
    Combine all the plots into a single figure using bash convert command
    This requires ImageMagick to be installed
    :param figure_type: str: type of the figure to combine (e.g., 'scatterplot', or 'diffplot')
    :param figure_fname_list
    :param output_dir: str: output directory where the combined figure will be saved
    :param panel_label: str: optional panel label (e.g., "A)", "B)") to add to the combined figure
    """

    # Create 'combined' directory if it does not exist
    combined_dir = os.path.join(output_dir, 'combined')
    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)

    print(f"Combining {figure_type}s into a single figure...")
    fname_out = os.path.join(combined_dir, f'combined_{figure_type}_subjects.png')

    # 1 row, 3 columns:
    cmd_combine = f"convert {figure_fname_list[0]} {figure_fname_list[1]} {figure_fname_list[2]} +append"

    # Add panel label if provided
    if panel_label:
        cmd_combine += f" -pointsize 80 -fill black -gravity NorthWest -annotate +20+20 '{panel_label}'"

    cmd_combine += f" {fname_out}"
    subprocess.run(cmd_combine, shell=True)
    print(f"Combined {figure_type} saved as {fname_out}")

    return fname_out


def create_scatterplot(df, output_dir):
    """
    Create scatter plots with linear regression lines for each metric
    :param df: pandas dataframe with lesion metrics
    :param output_dir: output directory
    """

    # Set font to Arial
    plt.rcParams['font.sans-serif'] = 'Arial'

    figure_fname_list = []

    for metric in METRIC_TO_TITLE.keys():
        df_plot = df[[f'{metric}_manual', f'{metric}_sct']]
        # Drop rows with NaN values
        df_plot = df_plot.dropna()

        fig, ax = plt.subplots(figsize=(6, 6))

        max_val = df_plot.max().max()
        min_val = df_plot.min().min()

        x = df_plot[f'{metric}_manual']
        y = df_plot[f'{metric}_sct']

        ax.scatter(x, y, s=40, alpha=0.7, color='black', edgecolor='black')
        ax.set_xlim(METRICS_AXIS_LIM_SCATTER[metric][0], METRICS_AXIS_LIM_SCATTER[metric][1])
        ax.set_ylim(METRICS_AXIS_LIM_SCATTER[metric][0], METRICS_AXIS_LIM_SCATTER[metric][1])

        # Add regression line
        intercept, slope, _, r2_sc, x_vals, y_vals = compute_regression(x, y)
        ax.plot(x_vals, y_vals, '-', color='black', linewidth=3)

        # Compute Spearman correlation
        spearman_corr, p_value = stats.spearmanr(x, y, nan_policy='omit')
        # # Compute paired test
        # stat, p_paired = stats.ttest_rel(x, y)
        ax.text(0.05, 0.95,
                # f'Spearman\nρ = {spearman_corr:.3f}\n{format_pvalue(p_value)}\nPaired test\n{format_pvalue(p_paired)}',
                f'ρ = {spearman_corr:.3f}\n{format_pvalue(p_value)}',
                transform=ax.transAxes, verticalalignment='top', fontsize=FONT_SIZE, color='black')

        # Add diagonal line
        ax.plot([METRICS_AXIS_LIM_SCATTER[metric][0], METRICS_AXIS_LIM_SCATTER[metric][1]],
                [METRICS_AXIS_LIM_SCATTER[metric][0], METRICS_AXIS_LIM_SCATTER[metric][1]],
                ls='--', c='gray', linewidth=2, alpha=0.7)

        # Change axes labels
        ax.set_title(f'{METRIC_TO_TITLE[metric]}', fontsize=FONT_SIZE)
        ax.set_xlabel(f'Manual', fontsize=FONT_SIZE)
        ax.set_ylabel(f'Automatic', fontsize=FONT_SIZE)

        if metric == 'midsagittal_length':
            # Tweak axes ticks
            ax.set_xticks([0, 50, 100, 150, 200])
            ax.set_yticks([0, 50, 100, 150, 200])
        elif metric == 'midsagittal_width':
            ax.set_xticks([0, 2, 4, 6, 8, 10])
            ax.set_yticks([0, 2, 4, 6, 8, 10])
        elif metric == 'total_tissue_bridge':
            ax.set_xticks([0, 2, 4, 6, 8])
            ax.set_yticks([0, 2, 4, 6, 8])
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)

        # Remove the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        # Save the plot
        num_subjects = len(df_plot)
        figure_fname = os.path.join(output_dir, f'scatterplot_{metric}_{num_subjects}subjects.png')
        figure_fname_list.append(figure_fname)
        plt.savefig(figure_fname, dpi=300)
        print(f'Scatter for {metric} saved as {figure_fname}')
        plt.close()

        # Add white space above figure
        cmd = f"convert '{figure_fname}' -bordercolor white -border 0x80+0+0 '{figure_fname}'"
        subprocess.run(cmd, shell=True, check=True)


    fname_out = combine_plot('scatterplot', figure_fname_list, output_dir, panel_label="A)")
    return fname_out


def create_diff_plot(df, output_dir):
    """
    Create a Bland-Altman Mean Difference Plot for each metric
    https://www.statsmodels.org/devel/generated/statsmodels.graphics.agreement.mean_diff_plot.html
    :param df: pandas dataframe with lesion metrics
    :param output_dir: output directory
    """

    # Set font to Arial
    plt.rcParams['font.sans-serif'] = 'Arial'

    figure_fname_list = []

    for metric in METRIC_TO_TITLE.keys():
        df_plot = df[[f'{metric}_manual', f'{metric}_sct']]

        # Control all font sizes for this plot in one place
        with plt.rc_context({
            'font.sans-serif': 'Arial',
            'font.size': FONT_SIZE,  # base font size
            'axes.labelsize': FONT_SIZE,  # x/y label size
            'axes.titlesize': FONT_SIZE,
            'xtick.labelsize': FONT_SIZE,
            'ytick.labelsize': FONT_SIZE,
            'legend.fontsize': FONT_SIZE
        }):
            fig, ax = plt.subplots(figsize=(6, 6))

        x = df_plot[f'{metric}_manual']
        y = df_plot[f'{metric}_sct']

        sm.graphics.mean_diff_plot(
            x, y,
            sd_limit=1.96,  # The default of 1.96 will produce 95% confidence intervals for the means of the differences
            ax=ax,
            scatter_kwds={
                's': 40,
                'alpha': 0.7,
                'color': 'black',
                'edgecolor': 'black',

            },
            mean_line_kwds={
                'color': 'black',
                'linestyle': '-',
                'alpha': 0.5,
                'linewidth': 3
            },
            limit_lines_kwds={
                'color': 'black',
                'linestyle': '--',
                'alpha': 0.5,
                'linewidth': 3
            }
        )

        ax.set_xlabel(f'Mean {METRIC_TO_TITLE[metric].split("[")[0]}', fontsize=FONT_SIZE)
        ax.set_ylabel(f'Difference Manual − Automatic',
                      fontsize=FONT_SIZE)

        # Get the limits and means for custom styling
        diff = x - y            # Difference between x and y
        sd = np.std(diff)       # Standard deviation of the difference
        mean_diff = np.mean(diff)

        # Calculate the actual limit values
        upper_limit = mean_diff + 1.96 * sd
        lower_limit = mean_diff - 1.96 * sd

        # Adjust y-lim
        # ax.set_ylim(-1.96 * sd * 1.5, 1.96 * sd * 1.5)
        ax.set_ylim(METRICS_AXIS_YLIM_DIFF[metric][0], METRICS_AXIS_YLIM_DIFF[metric][1])

        # Adjust x-lim
        ax.set_xlim(METRICS_AXIS_XLIM_DIFF[metric][0], METRICS_AXIS_XLIM_DIFF[metric][1])

        # Remove the default text labels that overlap with lines
        for t in ax.texts:
            t.remove()

        # Add custom positioned labels
        x_pos_right = ax.get_xlim()[1] * 0.95  # Position labels at 95% of x-axis

        # Add custom labels positioned away from the lines
        ax.text(x_pos_right, upper_limit + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03,
                f'+1.96 SD: {upper_limit:.1f}',
                horizontalalignment='right', verticalalignment='bottom',
                fontsize=FONT_SIZE, color='black')

        ax.text(x_pos_right, mean_diff + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03,
                f'mean diff: {mean_diff:.2f}',
                horizontalalignment='right', verticalalignment='bottom',
                fontsize=FONT_SIZE, color='black')

        ax.text(x_pos_right, lower_limit - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03,
                f'-1.96 SD: {lower_limit:.1f}',
                horizontalalignment='right', verticalalignment='top',
                fontsize=FONT_SIZE, color='black')

        # Optional: legend font size if present
        leg = ax.get_legend()
        if leg:
            for text in leg.get_texts():
                text.set_fontsize(FONT_SIZE)

        # Remove the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Draw dashed gray horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
        plt.tight_layout()

        # Save the plot
        num_subjects = len(df_plot)
        figure_fname = os.path.join(output_dir, f'diffplot_{metric}_{num_subjects}subjects.png')
        figure_fname_list.append(figure_fname)
        plt.savefig(figure_fname, dpi=300)
        print(f'Diffplot for {metric} saved as {figure_fname}')
        plt.close()

    fname_out = combine_plot(f'diffplot', figure_fname_list, output_dir, panel_label="B)")
    return fname_out

def main():
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()

    df = read_csv_file_with_lesion_metrics(args.i)

    # # Drop subjects with NaN values in any of the lesion metrics
    # df = df.dropna()
    # print(f'Number of subjects after dropping NaN values: {len(df)}')

    # Create output directory
    os.makedirs(args.o, exist_ok=True)

    # ----------------
    # Create scatter plots and diff plots for manual vs semi-automatic (manual lesions)
    # ----------------
    print('\nGenerating plots for Manual vs Automatic...')
    fname_combined_scatter = create_scatterplot(df, args.o)
    fname_combined_diff = create_diff_plot(df, args.o)

    print(f"Combining scatter and diff plots into a single figure...")
    fname_out = os.path.join(args.o, 'combined', f'Fig3_combined_scatter_and_diff.png')
    # Combine scatter and diff plots into a single figure (scatter on top, diff on bottom)
    cmd_combine = f"convert {fname_combined_scatter} {fname_combined_diff} -append {fname_out}"
    subprocess.run(cmd_combine, shell=True)
    print(f"Combined scatter and diff plots saved as {fname_out}")

if __name__ == '__main__':
    main()
