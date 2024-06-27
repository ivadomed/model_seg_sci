"""
Compare tissue bridges obtained using different methods (manual, semi-automatic, automatic)

The script:
 - computes normality test
 - computes Kruskal-Wallis H-test for the tissue bridges obtained using three methods:
    1. manual: manually segmented intramedullary lesions and manually measured tissue bridges
    2. semi-automatic: manually segmented intramedullary lesions and automatically measured tissue bridges
    3. automatic: automatically obtained intramedullary lesions using SCIsegV2 and automatically measured tissue bridges
 - creates pairplot showing relationships between three methods of tissue bridges measurements (manual, semi-automatic,
    automatic).

The data are stored in a CSV file with the following structure:
    | subject_id | dorsal_bridges | ventral_bridges | dorsal_bridges.1 | ventral_bridges.1 | dorsal_bridges.2 | ventral_bridges.2 |

Author: Jan Valosek
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from scipy.stats import normaltest, kruskal


def create_pairplot(df, csv_file, kind):
    """
    Create pairplot showing relationships between three methods of tissue bridges measurements (manual, semi-automatic,
    automatic).
    :param df: pandas dataframe with tissue bridges data
    :param csv_file: path to the input CSV file
    :param kind: ventral or dorsal
    :return:
    """
    # Keep only ventral or dorsal bridges columns
    df_plot = df[[kind+'_bridges', kind+'_bridges.1', kind+'_bridges.2']]
    # Rename columns: ventral_bridges to manual, ventral_bridges.1 to semi-auto, ventral_bridges.2 to auto
    df_plot.rename(columns={kind+'_bridges': 'manual',
                            kind+'_bridges.1': 'semi-automatic',
                            kind+'_bridges.2': 'automatic'},
                   inplace=True)

    # Plot sns.pairplot for ventral bridges
    g = sns.pairplot(df_plot[['manual', 'semi-automatic', 'automatic']], kind='reg',
                     plot_kws={'line_kws': {'color': 'black'}})
    # Set x and y-axis limits to -10% and +10% of the maximum value
    g.set(xlim=(-0.1 * df_plot.max().max(), 1.1 * df_plot.max().max()),
          ylim=(-0.1 * df_plot.max().max(), 1.1 * df_plot.max().max()))
    # Add title
    g.fig.suptitle(f'{kind.capitalize()} bridges [mm] obtained using different methods', y=0.99)
    plt.tight_layout()
    plt.show()
    # Save the plot into the same directory as the input CSV file
    figure_fname = os.path.join(os.path.dirname(csv_file), f'{kind}_bridges_pairplot.png')
    g.savefig(figure_fname, dpi=200)
    print(f'Pairplot for {kind} bridges saved as {figure_fname}')
    plt.close()


def main(csv_file):

    csv_file = os.path.expanduser(csv_file)
    # Read CSV file as a pandas dataframe, skip first row (header) and last row (empty row)
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f'File {csv_file} does not exist.')
    df = pd.read_csv(csv_file, skiprows=1, skipfooter=1)

    # Run normality test (D’Agostino and Pearson’s test) for each column
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
    for column in df.columns[1:]:
        stat, p = normaltest(df[column])
        print(f'Normality test for {column}: p-value = {p:.4f}')

    # Data does not have normal distribution --> performing Kruskal-Wallis H-tests
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
    stat, p = kruskal(df['dorsal_bridges'], df['dorsal_bridges.1'], df['dorsal_bridges.2'])
    print(f'Kruskal-Wallis H-test for dorsal bridges: p-value = {p:.4f}')
    stat, p = kruskal(df['ventral_bridges'], df['ventral_bridges.1'], df['ventral_bridges.2'])
    print(f'Kruskal-Wallis H-test for ventral bridges: p-value = {p:.4f}')

    # Create pairplot showing relationships between three methods of tissue bridges measurements (manual,
    # semi-automatic, automatic).
    create_pairplot(df, csv_file, 'ventral')
    create_pairplot(df, csv_file, 'dorsal')

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process CSV file for tissue bridges analysis.')
    parser.add_argument('-csv-file', type=str, help='Path to the CSV file', required=True)
    args = parser.parse_args()
    main(args.csv_file)

