import os
import re
import argparse
import seaborn as sns
from loguru import logger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test_sites = {
    "SCI": ["dcm-zurich-lesions-20231115", "sci-colorado", "sci-zurich", "site-003", "site-013", "site-014"],
    }

metrics_lesion = ['dsc', 'nsd', 'lesion_ppv', 'lesion_sensitivity', 'lesion_f1_score', 'lcwa']
metrics_sc = ['dsc', 'nsd']

def get_parser():

    parser = argparse.ArgumentParser(description='Plot metrics per site')
    parser.add_argument('-i', type=str, required=True, nargs='+',
                        help='Path to the folders containing all folds from nnUNet training for each model to compare')
    parser.add_argument('-o', type=str, required=True,
                        help='Path to the output folder where the plots will be saved')
    # parser.add_argument('--v2', type=str, required=True,
    #                     help='Path to the folder containing metrics for each site')
    parser.add_argument('-pathology', type=str, required=True,
                        help='Results to compare from which pathology (SCI or MS)')

    return parser

def find_site_in_path(path):
    """Extracts site identifier from the given path.

    Args:
    path (str): Input path containing a site identifier.

    Returns:
    str: Extracted site identifier or None if not found.
    """
    # Find 'dcm-zurich-lesions' or 'dcm-zurich-lesions-20231115'
    if 'dcm' in path:
        match = re.search(r'dcm-zurich-lesions(-\d{8})?', path)
    elif 'sci' in path:
        match = re.search(r'sci-zurich|sci-colorado|sci-paris', path)
    elif 'site' in path:
        # NOTE: PRAXIS data has 'site-xxx' in the path (and doesn't have the site names themselves in the path)
        match = re.search(r'site-\d{3}', path)

    return match.group(0) if match else None


def find_model_in_path(path):
    """Extracts model identifier from the given path.

    Args:
    path (str): Input path containing a model identifier.

    Returns:
    str: Extracted model identifier or None if not found.
    """
    # Find 'nnUNetTrainer' followed by the model name
    if 'Dataset501_allSCIsegV2RegionSeed710' in path:
        model = 'SCIsegV2Region'
    elif 'Dataset511_acuteSCIsegV2RegionSeed710' in path:
        model = 'acuteSCI'
    elif 'Dataset502_allSCIsegV2MultichannelSeed710' in path:
        model = 'SCIsegV2Multi'
    else:
        model = 'SCIsegV1'

    return model


def main():

    args = get_parser().parse_args()
    path_out = args.o
    if not os.path.exists(path_out):
        os.makedirs(path_out, exist_ok=True)

    num_models_to_compare = len(args.i)
    if num_models_to_compare < 2:
        raise ValueError("Please provide at least two models to compare")

    df_mega = pd.DataFrame()
    for fldr in args.i:

        df_folds = pd.DataFrame()
        # find folders using regex
        folds = sorted([f for f in os.listdir(fldr) if re.match(r'fold_\d\b', f)])

        for fld in folds:
            
            df_sites = pd.DataFrame()
            for site in test_sites["SCI"]:
                
                csv_file = os.path.join(fldr, fld, f'test_{site}', f'{site}_metrics_mean.csv')
                # print(f"Processing: {csv_file.replace('/home/GRAMES.POLYMTL.CA/u114716/', '~/')}")

                df = pd.read_csv(csv_file)
                df['site'] = site
                df['fold'] = fld
                df['model'] = find_model_in_path(fldr)

                # NOTE: because multi-channel model has only 1 label, it has to be renamed to 2.0 to match
                # the label id with the region-based models
                if df['model'].values[0] == 'SCIsegV2Multi':
                    df['label'] = 2.0

                df_sites = pd.concat([df_sites, df])

            df_folds = pd.concat([df_folds, df_sites])

        # print(df_folds)
        # compute the mean and std over all folds for each site
        # df_folds = df_folds.groupby(['label', 'site', 'model']).mean(numeric_only=True).reset_index()

        df_mega = pd.concat([df_mega, df_folds])
    
    # print(df_mega.reset_index(drop=True)))
    # print the df belong to model SCIsegV2Region
    # df_temp = df_mega[df_mega['model'] == 'SCIsegV2Region'].reset_index(drop=True)
    # print(df_temp)

    # compute the mean and std over sites for each model
    df_mega = df_mega.groupby(['model', 'site', 'label']).mean(numeric_only=True).reset_index()
    # print(df_mega.reset_index(drop=True))

    # # bring `model` and `site` columns to the front
    # cols = df_mega.columns.tolist()
    # cols = cols[-2:] + cols[:-2]
    # df_mega = df_mega[cols]

    print("Generating plots for Lesions")
    for metric in metrics_lesion:
        # keep the only the dataset, model, and metric columns
        df_metric = df_mega.copy()
        df_metric = df_metric[df_metric['label'] == 2.0]
        df_metric = df_metric[['model', 'site', f'{metric}_mean', f'{metric}_std']]

        # convert the metric values to float
        df_metric[f'{metric}_mean'] = df_metric[f'{metric}_mean'].astype(float)
        df_metric[f'{metric}_std'] = df_metric[f'{metric}_std'].astype(float)

        # use seaborn catplot to plot the metrics
        sns.set_theme(style="whitegrid")
        g = sns.catplot(
            data=df_metric, x='site', y=f'{metric}_mean',
            hue='model', kind='bar', aspect=2, alpha=0.6, height=6,
            # errorbar='sd'
        )
        # y-axis limits
        g.set(ylim=(0, 1))
        g.ax.set_yticks(np.arange(0, 1.1, 0.1))

        # add error bars from the std values in hte dataframe
        for bar, std in zip(g.ax.patches, df_metric[f'{metric}_std']):
            bar_height = bar.get_height()
            # plot with dashed lines
            g.ax.errorbar(
                bar.get_x() + bar.get_width() / 2,
                bar_height, 
                yerr=std, fmt='o', markersize=3,
                color='k', linewidth=1,
                capsize=3, capthick=1,
            )

        g.despine(left=True)
        g.set_axis_labels("Site", f"{metric}")
        g.legend.set_title("Model")

        print(f"\tSaving the plot for {metric}")
        plt.savefig(os.path.join(path_out, f"{metric}_lesion.png"))

    
    print("Generating plots for SC")
    for metric in metrics_sc:
        # keep the only the dataset, model, and metric columns
        df_metric = df_mega.copy()
        df_metric = df_metric[df_metric['label'] == 1.0]
        df_metric = df_metric[['model', 'site', f'{metric}_mean', f'{metric}_std']]

        # convert the metric values to float
        df_metric[f'{metric}_mean'] = df_metric[f'{metric}_mean'].astype(float)
        df_metric[f'{metric}_std'] = df_metric[f'{metric}_std'].astype(float)

        # use seaborn catplot to plot the metrics
        sns.set_theme(style="whitegrid")
        g = sns.catplot(
            data=df_metric, x='site', y=f'{metric}_mean',
            hue='model', kind='bar', aspect=2, alpha=0.6, height=6,
            # errorbar='sd'
        )
        # y-axis limits
        g.set(ylim=(0, 1))
        g.ax.set_yticks(np.arange(0, 1.1, 0.1))

        # add error bars from the std values in hte dataframe
        for bar, std in zip(g.ax.patches, df_metric[f'{metric}_std']):
            bar_height = bar.get_height()
            # plot with dashed lines
            g.ax.errorbar(
                bar.get_x() + bar.get_width() / 2,
                bar_height, 
                yerr=std, fmt='o', markersize=3,
                color='k', linewidth=1,
                capsize=3, capthick=1,
            )

        g.despine(left=True)
        g.set_axis_labels("Site", f"{metric}")
        g.legend.set_title("Model")

        print(f"\tSaving the plot for {metric}_sc")
        plt.savefig(os.path.join(path_out, f"{metric}_sc.png"))


if __name__ == '__main__':
    main()