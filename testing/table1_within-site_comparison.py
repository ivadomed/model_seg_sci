"""
Perform within-site age and sex comparison between test and train subjects.

Relevant for Table 1 from https://www.medrxiv.org/content/10.1101/2024.01.03.24300794v2.

The test/train subjects are listed in `dataset-conversion/dataset_split_seed710.yaml` file.
Note that we need to use the SCIsegV1 version of the dataset:
    https://github.com/ivadomed/model_seg_sci/blob/9832c82f25d2a7803c94cb53fb29303202cd1b31/dataset-conversion/dataset_split_seed710.yaml
Thus, I downloaded the file and passed the path to the script as an argument ('-yml-file').

Example usage:
    python table1_within-site_comparison.py -yml-file ~/Downloads/dataset_split_seed710.yaml

Author: Jan Valosek
"""


import os
import argparse

import numpy as np
import pandas as pd
import yaml     # pip install pyyaml

from scipy.stats import normaltest, mannwhitneyu, chi2_contingency


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='Perform within-site age and sex comparison between test and train subjects.',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-yml-file',
        metavar="<file>",
        required=True,
        type=str,
        help='Path to the YML file listing train and test subjects. '
             'Example: dataset_split_seed710.yaml'
    )

    return parser


def read_yaml_file(file_path, key):
    """
    Read YAML file
    Args:
        file_path: Path to the YAML file
        key: Key to fetch from the JSON file, e.g., train, test

    Returns:
        list of subjects with surgery
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    # Check if the key exists in the YAML file
    if key not in data:
        raise ValueError(f'ERROR: {key} does not exist in {file_path}')

    subject_list = data[key]

    # Keep only participant_id (e.g., sub-6577), i.e., remove everything after the first underscore
    subject_list = [subject.split('_')[0] for subject in subject_list]

    return subject_list


def read_participants_tsv(dataset_name):
    """
    Read participant's age from participants.tsv files for sci-zurich and sci-colorado
    :param dataset_name: Name of the dataset (e.g., sci-zurich, sci-colorado)
    :return: DataFrame with participant's age and sex
    """
    datasets_path = '~/data/data.neuro.polymtl.ca'
    participants_tsv = pd.read_csv(f'{datasets_path}/{dataset_name}/participants.tsv',
                                   usecols=['participant_id', 'age', 'sex'],
                                   sep='\t')

    # Add a new column 'site' with the dataset name
    participants_tsv['site'] = dataset_name

    return participants_tsv


def compare_age(df):
    """
    For each site, perform within-site age comparison between test and train subjects
    First, we test data normality using D’Agostino and Pearson’s normality test
    If the data is NOT normally distributed, we use the Mann-Whitney U test (non-parametric test, independent samples)
    :param df:
    :return:
    """
    alpha = 0.05
    for site in ['sci-zurich', 'sci-colorado', 'both']:

        # if site is 'both', we combine subjects from both sites
        if site == 'both':
            site_df = df
        else:
            site_df = df[df['site'] == site]

        train_age = site_df[site_df['train_test'] == 'train']['age']
        test_age = site_df[site_df['train_test'] == 'test']['age']

        print(f"{site}, train subjects: {train_age.mean():.2f} +- {train_age.std():.2f} years")
        print(f"{site}, test subjects: {test_age.mean():.2f} +- {test_age.std():.2f} years")

        # Test data normality
        stat, p_value = normaltest(train_age)
        print(f"{site}, train data: normality test p-value: {p_value:.4f}")
        if p_value > alpha:
            print(f'{site}, train data is normally distributed')
        else:
            print(f'{site}, train data is NOT normally distributed')

        stat, p_value = normaltest(test_age)
        print(f"{site}, test data: normality test p-value: {p_value:.4f}")
        if p_value > alpha:
            print(f'{site}, test data is normally distributed')
        else:
            print(f'{site}, test data is NOT normally distributed')

        # Perform Mann-Whitney U test
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
        # H0: the distribution underlying sample x is the same as the distribution underlying sample y
        stat, p_value = mannwhitneyu(train_age, test_age)
        print(f"{site}: Mann-Whitney U test p-value: {p_value:.4f}")
        if p_value > alpha:
            print(f'{site}: NO significant difference (same distribution) --> fail to reject H0.')
        else:
            print(f'{site}: A significant difference (different distribution) --> reject H0.')


def compare_sex_ratios(df):
    """
    For each site, compare the sex ratios between test and train subjects using the chi-squared test for independence.
    """
    alpha = 0.05
    # Create a contingency table
    for site in ['sci-zurich', 'sci-colorado', 'both']:

        # if site is 'both', we combine subjects from both sites
        if site == 'both':
            site_df = df
        else:
            site_df = df[df['site'] == site]

        # Get a tuple containing the counts of females and males
        train_counts = (len(site_df[(site_df['train_test'] == 'train') & (site_df['sex'] == 'F')]),
                        len(site_df[(site_df['train_test'] == 'train') & (site_df['sex'] == 'M')]))
        test_counts = (len(site_df[(site_df['train_test'] == 'test') & (site_df['sex'] == 'F')]),
                       len(site_df[(site_df['train_test'] == 'test') & (site_df['sex'] == 'M')]))

        print(f'{site}, train counts (F/M): {train_counts}')
        print(f'{site}, test counts (F/M): {test_counts}')

        # Create a contingency table
        contingency_table = np.array([train_counts, test_counts])

        # Perform the chi-squared test
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        print(f"{site}: chi-squared test p-value: {p_value:.4f}")
        if p_value > alpha:
            print(f'{site}: NO significant difference in sex ratios between the two groups --> fail to reject H0.')
        else:
            print(f'{site}: A significant difference in sex ratios between the two groups --> reject H0.')


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    yml_file = os.path.abspath(os.path.expanduser(args.yml_file))

    if not os.path.isfile(yml_file):
        print(f'ERROR: {yml_file} does not exist.')

    # Get the list of test and train subjects
    test_subjects = read_yaml_file(yml_file, key='test')
    train_subjects = read_yaml_file(yml_file, key='train')

    # Get participant's age and sex from participants.tsv files for sci-zurich and sci-colorado
    # The participants.tsv files are located in the BIDS dataset folder
    # Read 'participant_id', 'age', and 'sex columns
    zurich_df = read_participants_tsv('sci-zurich')
    colorado_df = read_participants_tsv('sci-colorado')

    # Keep only subjects from the test and train lists
    zurich_df = zurich_df[zurich_df['participant_id'].isin(test_subjects + train_subjects)]
    colorado_df = colorado_df[colorado_df['participant_id'].isin(test_subjects + train_subjects)]

    # Merge all dataframes into one, adding a column with train/test label
    df = pd.concat([zurich_df, colorado_df])
    df['train_test'] = 'train'
    df.loc[df['participant_id'].isin(test_subjects), 'train_test'] = 'test'

    # Some subjects have nan age, drop them
    df.dropna(subset=['age'], inplace=True)

    # For each site, perform within-site age comparison between test and train subjects
    compare_age(df)
    # For each site, compare the sex ratios between test and train subjects
    compare_sex_ratios(df)


if __name__ == '__main__':
    main()
