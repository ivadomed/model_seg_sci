"""
Compare the midsagittal lesion length, lesion width, and tissue bridges obtained using different methods (manual,
automatic SCT master, automatic SCT PR4631).

The script:
 - reads XLS files with manually measured lesion metrics and CSV files with lesion metrics computed using sct_analyze_lesion
- computes Wilcoxon signed-rank test for each metric between methods
- creates pairplot showing relationships between methods
- creates Bland-Altman Mean Difference Plot for each metric

Note: to read XLS files, you might need to install the following packages:
    pip install openpyxl

Author: Jan Valosek
"""
import numpy as np

METRICS = ['midsagittal_length', 'midsagittal_width', 'ventral_tissue_bridge', 'dorsal_tissue_bridge']
METRIC_TO_TITLE = {
    'midsagittal_length': 'Midsagittal Lesion Length [mm]',
    'midsagittal_width': 'Midsagittal Lesion Width [mm]',
    'ventral_tissue_bridge': 'Midsagittal Ventral Tissue Bridges [mm]',
    'dorsal_tissue_bridge': 'Midsagittal Dorsal Tissue Bridges [mm]'
}

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from sklearn.linear_model import LinearRegression
from scipy.stats import wilcoxon
import statsmodels.api as sm


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='Read CSV files with lesion metrics computed using sct_analyze_lesion and XLSX file with manually'
                    'measured metrics.',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-manual',
        required=True,
        type=str,
        help='Absolute path to the XLSX file with manually measured lesion metrics'
    )
    parser.add_argument(
        '-master',
        required=True,
        type=str,
        help='Absolute path to the CSV file with lesion metrics computed using sct_analyze_lesion on the master branch'
    )
    parser.add_argument(
        '-PR4631',
        required=True,
        type=str,
        help='Absolute path to the CSV file with lesion metrics computed using sct_analyze_lesion on the PR4631 branch'
    )
    parser.add_argument(
        '-o',
        required=False,
        default='stats/figures',
        help='Path to the output folder where XLS table will be saved. Default: ./stats'
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


def create_pairplot(df, output_dir):
    """
    Create pairplot showing relationships between methods in a 2x2 grid
    :param df: pandas dataframe with metrics data
    :param output_dir: output directory
    :return:
    """
    for metric in METRICS:
        df_plot = df[[f'{metric}_manual', f'{metric}_master', f'{metric}_PR4631']]
        # Rename columns for better axis labels
        df_plot.rename(columns={metric + '_manual': 'Manual',
                                metric + '_master': 'SCT master',
                                metric + '_PR4631': 'SCT PR4631'},
                       inplace=True)

        fig, axes = plt.subplots(2, 2, figsize=(6, 6))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        max_val = df_plot.max().max()
        min_val = df_plot.min().min()

        # Define the plots we want to show
        plots = [(0, 0, 'Manual', 'SCT master'),
                 (1, 0, 'Manual', 'SCT PR4631'),
                 (1, 1, 'SCT master', 'SCT PR4631')]

        for row, col, x_method, y_method in plots:
            ax = axes[row, col]
            x = df_plot[x_method]
            y = df_plot[y_method]

            ax.scatter(x, y)
            ax.set_xlim(-0.1 * max_val, 1.1 * max_val)
            ax.set_ylim(-0.1 * max_val, 1.1 * max_val)

            ax.set_xlabel(x_method)
            ax.set_ylabel(y_method)

            # Add regression line
            intercept, slope, _, r2_sc, x_vals, y_vals = compute_regression(x, y)
            ax.plot(x_vals, y_vals, '--', color='red')

            # Add R² value to the plot
            ax.text(0.05, 0.95, f'R² = {r2_sc:.2f}', transform=ax.transAxes,
                    verticalalignment='top')

            # Add diagonal line
            ax.plot([min_val, max_val], [min_val, max_val], ls='--', c='gray')

        # Turn off the unused subplot
        axes[0, 1].axis('off')

        # Add title
        fig.suptitle(f'{METRIC_TO_TITLE[metric]}', y=0.99)
        plt.tight_layout()

        # Save the plot
        figure_fname = os.path.join(output_dir, f'{metric}.png')
        plt.savefig(figure_fname, dpi=200)
        print(f'Pairplot for {metric} bridges saved as {figure_fname}')
        plt.close()


def create_diff_plot(df, output_dir):
    """
    Create a Bland-Altman Mean Difference Plot for each metric
    https://www.statsmodels.org/devel/generated/statsmodels.graphics.agreement.mean_diff_plot.html
    :param df:
    :param output_dir:
    :return:
    """
    for metric in METRICS:
        df_plot = df[[f'{metric}_manual', f'{metric}_master', f'{metric}_PR4631']]
        # Rename columns for better axis labels
        df_plot.rename(columns={metric + '_manual': 'Manual',
                                metric + '_master': 'SCT master',
                                metric + '_PR4631': 'SCT PR4631'},
                       inplace=True)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        # Define the plots we want to show
        plots = [(0, 'Manual' , 'SCT master'),
                 (1, 'Manual', 'SCT PR4631')]

        for col, x_method, y_method in plots:
            ax = axes[col]
            x = df_plot[x_method]
            y = df_plot[y_method]
            mean_diff_plot = sm.graphics.mean_diff_plot(x, y, ax=ax)
            ax.set_title(f'{METRIC_TO_TITLE[metric]}\n{x_method} vs {y_method}')

        plt.tight_layout()

        # Save the plot
        figure_fname = os.path.join(output_dir, f'{metric}_diffplot.png')
        plt.savefig(figure_fname, dpi=200)
        print(f'Diffplot for {metric} bridges saved as {figure_fname}')
        plt.close()



def main():

    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()
    output_dir = args.o
    # create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    xlsx_manual = args.manual
    csv_master = args.master
    csv_PR4631 = args.PR4631

    # MANUAL
    df_manual = pd.read_excel(xlsx_manual)
    # If session_id is nan in the manual file, set it to 'ses-01'
    df_manual['session_id'] = df_manual['session_id'].fillna('ses-01')
    # Add '_manual' suffix to all columns except participant_id and session_id
    df_manual = df_manual.add_suffix('_manual')
    df_manual.rename(columns={'participant_id_manual': 'participant_id', 'session_id_manual': 'session_id'}, inplace=True)

    # MASTER BRANCH
    df_master = pd.read_csv(csv_master)
    # Add '_master' suffix to all columns except participant_id and session_id
    df_master = df_master.add_suffix('_master')
    df_master.rename(columns={'participant_id_master': 'participant_id', 'session_id_master': 'session_id'}, inplace=True)

    # PR4631 BRANCH
    df_PR4631 = pd.read_csv(csv_PR4631)
    # Add '_PR4631' suffix to all columns except participant_id and session_id
    df_PR4631 = df_PR4631.add_suffix('_PR4631')
    df_PR4631.rename(columns={'participant_id_PR4631': 'participant_id', 'session_id_PR4631': 'session_id'}, inplace=True)

    # Merge the dataframes
    df = pd.merge(df_manual, df_master, on=['participant_id', 'session_id'])
    df = pd.merge(df, df_PR4631, on=['participant_id', 'session_id'])

    # Replace nan values with zeros (if there is no lesion, we assume the metrics are zero)
    df = df.fillna(0)

    # Print number of subjects (rows)
    print(f'Number of subjects: {df.shape[0]}')

    # Print participant_id presented in df_manual but not in df_master
    missing_participant_id = df_manual[~df_manual['participant_id'].isin(df_master['participant_id'])]['participant_id']
    if not missing_participant_id.empty:
        print('participant_id presented in df_manual but not in df_master:')
        print(missing_participant_id)

    # Compute Wilcoxon signed-rank test for each metric between methods
    for metric in METRICS:
        # Compute Wilcoxon signed-rank test
        _, p_value_master = wilcoxon(df[f'{metric}_manual'], df[f'{metric}_master'])
        _, p_value_PR4631 = wilcoxon(df[f'{metric}_manual'], df[f'{metric}_PR4631'])

        print(f'Wilcoxon signed-rank test for {METRIC_TO_TITLE[metric]}:')
        print(f'  - p-value (manual vs master): {p_value_master:.3f}')
        print(f'  - p-value (manual vs PR4631): {p_value_PR4631:.3f}')

    # Create pairplot showing relationships between methods
    #create_pairplot(df, output_dir)
    create_diff_plot(df, output_dir)


if __name__ == '__main__':
    main()


