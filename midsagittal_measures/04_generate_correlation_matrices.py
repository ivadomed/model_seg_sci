"""
Generate correlation matrices between three methods for each lesion metric.

The script:
- reads XLSX file with manually measured lesion metrics
- reads CSV files with lesion metrics computed using sct_analyze_lesion for both GT and SCIsegV2 methods
- merges the dataframes
- creates correlation matrices for each metric showing correlations between manual, semi-automatic (GT), and automatic (SCIsegV2) methods
- computes both Pearson and Spearman correlations with statistical significance

Example usage:
    python 04_generate_correlation_matrices.py
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
from scipy import stats


METRIC_TO_TITLE = {
    'midsagittal_length': 'Midsagittal Lesion Length [mm]',
    'midsagittal_width': 'Midsagittal Lesion Width [mm]',
    'ventral_tissue_bridge': 'Midsagittal Ventral Tissue Bridges [mm]',
    'dorsal_tissue_bridge': 'Midsagittal Dorsal Tissue Bridges [mm]',
    'total_tissue_bridge': 'Midsagittal Total Tissue Bridges [mm]',
    'dorsal_bridge_ratio': 'Midsagittal Dorsal Tissue Bridge Ratio [%]',
    'ventral_bridge_ratio': 'Midsagittal Ventral Tissue Bridge Ratio [%]',
}

FONT_SIZE = 10


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
    return df_manual


def read_file_sct(file_sct, method_suffix):
    """
    Read CSV file with lesion metrics computed using sct_analyze_lesion.
    :param file_sct: str: path to the CSV file
    :param method_suffix: str: suffix to add to metric columns (e.g., 'gt' or 'scisegv2')
    :return df_sct: pandas DataFrame: dataframe with SCT-computed lesion metrics
    """
    df_sct = pd.read_csv(file_sct)
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
    method_labels = ['Manual', 'Semi-automatic\n(GT + SCT)', 'Automatic\n(SCIsegV2 + SCT)']

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

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pearson correlation matrix
    mask_pearson = np.triu(np.ones_like(pearson_corr, dtype=bool))
    sns.heatmap(pearson_corr, mask=mask_pearson, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                xticklabels=method_labels, yticklabels=method_labels,
                vmin=-1, vmax=1, fmt='.3f', ax=ax1, annot_kws={'size': FONT_SIZE})
    ax1.set_title(f'Pearson Correlation\n{METRIC_TO_TITLE[metric]}', fontsize=FONT_SIZE + 2)

    # Add p-values as text annotations
    for i in range(n_methods):
        for j in range(n_methods):
            if i > j:  # Lower triangle only
                p_val = pearson_pvals[i, j]
                if not np.isnan(p_val):
                    ax1.text(j + 0.5, i + 0.75, format_pvalue(p_val),
                            ha='center', va='center', fontsize=FONT_SIZE - 2, color='black')

    # Spearman correlation matrix
    mask_spearman = np.triu(np.ones_like(spearman_corr, dtype=bool))
    sns.heatmap(spearman_corr, mask=mask_spearman, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                xticklabels=method_labels, yticklabels=method_labels,
                vmin=-1, vmax=1, fmt='.3f', ax=ax2, annot_kws={'size': FONT_SIZE})
    ax2.set_title(f'Spearman Correlation\n{METRIC_TO_TITLE[metric]}', fontsize=FONT_SIZE + 2)

    # Add p-values as text annotations
    for i in range(n_methods):
        for j in range(n_methods):
            if i > j:  # Lower triangle only
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

    # Print correlation summary to console
    print(f'\n--- {METRIC_TO_TITLE[metric]} ({num_subjects} subjects) ---')
    print('Pearson correlations:')
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            corr_val = pearson_corr.iloc[i, j]
            p_val = pearson_pvals[i, j]
            print(f'  {method_labels[i]} vs {method_labels[j]}: r = {corr_val:.3f}, {format_pvalue(p_val)}')

    print('Spearman correlations:')
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            corr_val = spearman_corr.iloc[i, j]
            p_val = spearman_pvals[i, j]
            print(f'  {method_labels[i]} vs {method_labels[j]}: ρ = {corr_val:.3f}, {format_pvalue(p_val)}')


def main():
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Read the data files
    print('Reading data files...')
    df_manual = read_file_manual(args.file_manual)
    df_gt = read_file_sct(args.file_gt, 'gt')
    df_scisegv2 = read_file_sct(args.file_scisegv2, 'scisegv2')

    # Merge the dataframes
    print('Merging dataframes...')
    df = pd.merge(df_manual, df_gt, on=['participant_id', 'session_id'], how='inner')
    df = pd.merge(df, df_scisegv2, on=['participant_id', 'session_id'], how='inner')

    # Keep only ses-01 sessions
    df_ses_01 = df[df['session_id'] == 'ses-01']
    print(f'Number of subjects with ses-01 data: {len(df_ses_01)}')

    # Filter subjects based on MRI time since injury (if available)
    if 'mri_time_since_injury' in df_ses_01.columns:
        df_ses_01['mri_time_since_injury'] = pd.to_numeric(df_ses_01['mri_time_since_injury'], errors='coerce')
        print(f'Number of subjects before filtering by MRI time since injury: {df_ses_01.shape[0]}')
        # Keep only subjects with mri_time_since_injury (in days) from 12 days to 2 months
        df_ses_01 = df_ses_01[(df_ses_01['mri_time_since_injury'] >= 12) &
                              (df_ses_01['mri_time_since_injury'] <= 60)]
        print(f'Number of subjects after filtering by MRI time since injury: {df_ses_01.shape[0]}')

    # Create output directory
    os.makedirs(args.o, exist_ok=True)

    # Generate correlation matrices for each metric
    print('\nGenerating correlation matrices...')
    for metric in METRIC_TO_TITLE.keys():
        create_correlation_matrix(df_ses_01, metric, args.o)

    print(f'\nAll correlation matrices saved to: {args.o}')


if __name__ == '__main__':
    main()
