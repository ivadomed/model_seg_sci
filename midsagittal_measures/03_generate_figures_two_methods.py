"""
Compare the lesion metrics obtained using different methods.

The script:
- reads XLS (manually measured lesion metrics) or CSV files (metrics computed using sct_analyze_lesion) for two methods
- creates scatter plots with linear regression lines for each metric
- creates Bland-Altman Mean Difference plot for each metric

Example usage:
    python 03_generate_figures_two_methods.py
        -file1 lesion_metrics_manual.xlsx -method1 manual
        -file2 lesion_metrics_SCIsegV2_PR4631.csv -method2 SCIsegV2_PR4631


Note: to read XLS files, you might need to install the following packages:
    pip install openpyxl

Author: Jan Valosek
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


METRICS = ['midsagittal_length', 'midsagittal_width', 'ventral_tissue_bridge', 'dorsal_tissue_bridge']
METRIC_TO_TITLE = {
    'length': '3D Lesion Length [mm]',
    'width': '3D Lesion Width [mm]',
    'midsagittal_length': 'Midsagittal Lesion Length [mm]',
    'midsagittal_width': 'Midsagittal Lesion Width [mm]',
    'ventral_tissue_bridge': 'Midsagittal Ventral Tissue Bridges [mm]',
    'dorsal_tissue_bridge': 'Midsagittal Dorsal Tissue Bridges [mm]'
}
METHOD_TO_AXIS = {
    'manual': 'Manual measurements',
    'GT': 'Automatic measurements (Ground Truth)',
    'SCIsegV2': 'Automatic measurements (SCIsegV2)',
}

def get_method_key(method):
    if method.startswith('GT_'):
        return 'GT'
    elif method.startswith('SCIsegV2_'):
        return 'SCIsegV2'
    elif method.startswith('manual'):
        return 'manual'


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
        '-file1',
        required=True,
        type=str,
        help='Absolute path to a CSV/XLSX file with lesion metrics for method 1.'
    )
    parser.add_argument(
        '-file2',
        required=True,
        type=str,
        help='Absolute path to a CSV/XLSX file with lesion metrics for method 2.'
    )
    parser.add_argument(
        '-method1',
        required=True,
        type=str,
        help='Name of method 1 (e.g., manual).'
    )
    parser.add_argument(
        '-method2',
        required=True,
        type=str,
        help='Name of method 2 (e.g., GT_master, GT_PR4631, SCIsegV2_PR4631).'
    )
    parser.add_argument(
        '-o',
        required=False,
        default='stats/figures',
        help='Path to the output folder where XLS table will be saved. Default: ./stats'
    )

    return parser


def read_xlsx(file):
    """
    Read XLSX file with manually measured metrics
    :param file: str: path to the XLSX file
    :return: pd.DataFrame: dataframe with the metrics
    """
    df = pd.read_excel(file)
    # If session_id is nan in the manual file, set it to 'ses-01'
    df['session_id'] = df['session_id'].fillna('ses-01')
    return df


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


def create_scatterplot(df, method1, method2, output_dir):
    """
    Create scatter plots with linear regression lines for each metric
    :param df: pandas dataframe with metrics data
    :param method1: name of the first method
    :param method2: name of the second method
    :param output_dir: output directory
    """

    # Set font to Arial
    plt.rcParams['font.sans-serif'] = 'Arial'

    for metric in METRICS:
        df_plot = df[[f'{metric}_{method1}', f'{metric}_{method2}']]

        fig, axes = plt.subplots(figsize=(5, 5))

        max_val = df_plot.max().max()
        min_val = df_plot.min().min()

        ax = axes
        x = df_plot[f'{metric}_{method1}']
        y = df_plot[f'{metric}_{method2}']

        ax.scatter(x, y, s=90, alpha=0.5)
        ax.set_xlim(-0.1 * max_val, 1.1 * max_val)
        ax.set_ylim(-0.1 * max_val, 1.1 * max_val)

        # Add regression line
        intercept, slope, _, r2_sc, x_vals, y_vals = compute_regression(x, y)
        ax.plot(x_vals, y_vals, '--', color='red')

        # Add R² value to the plot
        ax.text(0.05, 0.95, f'R² = {r2_sc:.2f}', transform=ax.transAxes,
                verticalalignment='top', fontsize=15, color='red')

        # Add diagonal line
        ax.plot([min_val, max_val], [min_val, max_val], ls='--', c='gray')

        # Change axes labels
        ax.set_xlabel(f'{METHOD_TO_AXIS[get_method_key(method1)]}', fontsize=15)
        ax.set_ylabel(f'{METHOD_TO_AXIS[get_method_key(method2)]}', fontsize=15)

        if metric == 'midsagittal_length':
            # Change axes ticks to 0, 50, 100, 150, 200
            ax.set_xticks([0, 50, 100, 150, 200])
            ax.set_yticks([0, 50, 100, 150, 200])

        plt.tight_layout()

        # Remove the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Save the plot
        figure_fname = os.path.join(output_dir, f'{metric}_{method1}_{method2}_scatterplot.png')
        plt.savefig(figure_fname, dpi=300)
        print(f'Pairplot for {metric} saved as {figure_fname}')
        plt.close()


def create_scatterplot_3D_length_width(df, method1, method2, output_dir):
    """
    Create scatter plots with linear regression lines for each metric
        - between 3D length and manual midsagittal length
        - between 3D width and manual midsagittal width
    :param df: pandas dataframe with metrics data
    :param method1: name of the first method
    :param method2: name of the second method
    :param output_dir: output directory
    """

    # Set font to Arial
    plt.rcParams['font.sans-serif'] = 'Arial'

    for metric in ['length', 'width']:
        df_plot = df[[f'midsagittal_{metric}_{method1}', f'{metric}_{method2}']]

        fig, axes = plt.subplots(figsize=(5, 5))

        max_val = df_plot.max().max()
        min_val = df_plot.min().min()

        ax = axes
        x = df_plot[f'midsagittal_{metric}_{method1}']
        y = df_plot[f'{metric}_{method2}']

        ax.scatter(x, y)
        ax.set_xlim(-0.1 * max_val, 1.1 * max_val)
        ax.set_ylim(-0.1 * max_val, 1.1 * max_val)

        # Add regression line
        intercept, slope, _, r2_sc, x_vals, y_vals = compute_regression(x, y)
        ax.plot(x_vals, y_vals, '--', color='red')

        # Add R² value to the plot
        ax.text(0.05, 0.95, f'R² = {r2_sc:.2f}', transform=ax.transAxes,
                verticalalignment='top', fontsize=15, color='red')

        # Add diagonal line
        ax.plot([min_val, max_val], [min_val, max_val], ls='--', c='gray')

        # Change axes labels
        ax.set_xlabel(f'{METHOD_TO_AXIS[get_method_key(method1)]}', fontsize=15)
        ax.set_ylabel(f'{METHOD_TO_AXIS[get_method_key(method2)]}', fontsize=15)

        if metric == 'length':
            # Change axes ticks to 0, 50, 100, 150, 200
            ax.set_xticks([0, 50, 100, 150, 200])
            ax.set_yticks([0, 50, 100, 150, 200])

        plt.tight_layout()

        # Save the plot
        figure_fname = os.path.join(output_dir, f'{metric}_{method1}_{method2}_scatterplot.png')
        plt.savefig(figure_fname, dpi=200)
        print(f'Pairplot for 3D {metric} saved as {figure_fname}')
        plt.close()


def create_diff_plot(df, method1, method2, output_dir):
    """
    Create a Bland-Altman Mean Difference Plot for each metric
    https://www.statsmodels.org/devel/generated/statsmodels.graphics.agreement.mean_diff_plot.html
    :param df: pandas dataframe with metrics data
    :param method1: name of the first method
    :param method2: name of the second method
    :param output_dir: output directory
    """

    # Set font to Arial
    plt.rcParams['font.sans-serif'] = 'Arial'

    for metric in METRICS:
        df_plot = df[[f'{metric}_{method1}', f'{metric}_{method2}']]

        fig, axes = plt.subplots(figsize=(5, 5))

        ax = axes
        x = df_plot[f'{metric}_{method1}']
        y = df_plot[f'{metric}_{method2}']

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
        #ax.set_title(f'{METRIC_TO_TITLE[metric]}', fontsize=15)
        ax.set_xlabel(f'Mean of Manual and Automatic', fontsize=15)
        ax.set_ylabel(f'Difference of Manual and Automatic', fontsize=15)

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
        figure_fname = os.path.join(output_dir, f'{metric}_{method1}_{method2}_diffplot.png')
        plt.savefig(figure_fname, dpi=300)
        print(f'Diffplot for {metric} bridges saved as {figure_fname}')
        plt.close()



def main():

    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()
    output_dir = args.o
    # create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the data
    file1 = args.file1
    file2 = args.file2
    method1 = args.method1
    method2 = args.method2

    #----------------
    # Method 1
    #----------------
    # XLSX is available only for manual measurements
    if file1.endswith('.xlsx'):
        df_method1 = read_xlsx(file1)
    elif file1.endswith('.csv'):
        df_method1 = pd.read_csv(file1)

    # Add suffix to all columns except participant_id and session_id
    df_method1 = df_method1.add_suffix(f'_{method1}')
    df_method1.rename(columns={'participant_id_' + method1: 'participant_id',
                               'session_id_' + method1: 'session_id'}, inplace=True)

    #----------------
    # Method 2
    #----------------
    if file2.endswith('.xlsx'):
        df_method2 = read_xlsx(file2)
    elif file2.endswith('.csv'):
        df_method2 = pd.read_csv(file2)

    # Add suffix to all columns except participant_id and session_id
    df_method2 = df_method2.add_suffix(f'_{method2}')
    df_method2.rename(columns={'participant_id_' + method2: 'participant_id',
                               'session_id_' + method2: 'session_id'}, inplace=True)

    # Merge the dataframes
    df = pd.merge(df_method1, df_method2, on=['participant_id', 'session_id'])

    # Replace nan values with zeros (if there is no lesion, we assume the metrics are zero)
    df = df.fillna(0)

    # Print number of subjects (rows)
    print(f'Number of subjects: {df.shape[0]}')

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

    # Exclude sub-zh15, sub-zh81
    df = df[~df['participant_id'].isin(['sub-zh15', 'sub-zh81'])]
    print(len(df))

    # Print participant_ids for subjects with high midsagittal_length > 100 mm
    print(f'Subjects with midsagittal_length_{args.method2} > 100 mm')
    print(df[df[f'midsagittal_length_{args.method2}'] > 100][['participant_id', 'session_id']])

    # Create scatter plots with linear regression lines
    #create_scatterplot(df, method1, method2, output_dir)
    # Create scatter plots for 3D lesion length and width
    #create_scatterplot_3D_length_width(df, method1, method2, output_dir)
    create_diff_plot(df, method1, method2, output_dir)

    # Keep only participant_id, session_id and midsagittal_slice columns
    df_to_save = df[['participant_id', 'session_id', 'midsagittal_slice_' + method2]]
    # Save the dataframe with the midsagittal slice to a CSV file
    df_to_save.to_csv(os.path.join(output_dir, f'midsagittal_slice_{method2}.csv'), index=False)


if __name__ == '__main__':
    main()
