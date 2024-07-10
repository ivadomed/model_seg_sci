"""
This script generates bar plots for the selected metrics across all sites and models. It automatically
iterates through all the available folds, plots the metrics and also saves them into a CSV file. 
The output CSV file is then used to generate the lesion-wise metrics table for the SCIsegV2 paper.
"""
import os
import re
import argparse
import seaborn as sns
from loguru import logger

import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

test_sites = {
    "SCI": ["dcm-zurich-lesions-20231115", "sci-colorado", "sci-zurich", "site-003", "site-014"],
    }

sites_to_rename = {
  'dcm-zurich-lesions-20231115': 'site-01 \n(non-traumatic SCI)',
  'sci-colorado': 'site-02 \n(traumatic SCI)',
  'sci-zurich': 'site-01 \n(traumatic SCI)',
  'sci-paris': 'site-03',                           # not shown in the figure (training only), but listing here for completeness
  'site-003': 'site-04 \n(acute traumatic SCI)',
  'site-012': 'site-05 \n(acute traumatic SCI)',    # not shown in the figure (training only), but listing here for completeness
  'site-013': 'site-06 \n(acute traumatic SCI)',
  'site-014': 'site-07 \n(acute traumatic SCI)',
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
        if 'nnUNetTrainer__nnUNetPlans__3d_fullres' in path:
            model = 'SCIsegV2_single\ndefaultDA'
        elif 'nnUNetTrainerDA5__nnUNetPlans__3d_fullres' in path:
            model = 'SCIsegV2_single\naggressiveDA'
    
    elif 'Dataset502_allSCIsegV2MultichannelSeed710' in path:
        if 'nnUNetTrainer__nnUNetPlans__3d_fullres' in path:
            model = 'SCIsegV2_multi\ndefaultDA'
        elif 'nnUNetTrainerDA5__nnUNetPlans__3d_fullres' in path:
            model = 'SCIsegV2_multi\naggressiveDA'
    
    elif 'Dataset521_DCMsegV2RegionSeed710' in path:
        model = 'DCM'
    
    elif 'Dataset511_acuteSCIsegV2RegionSeed710' in path:
        model = 'AcuteSCI'
    
    else:
        model = 'SCIsegV1'

    return model


def main():

    args = get_parser().parse_args()
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    path_out = f"{args.o}" #_{now}"
    if not os.path.exists(path_out):
        os.makedirs(path_out, exist_ok=True)

    num_models_to_compare = len(args.i)
    if num_models_to_compare < 2:
        raise ValueError("Please provide at least two models to compare")

    num_subs_per_site = {}
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
                df['site'] = sites_to_rename[site]
                df['fold'] = fld
                df['model'] = find_model_in_path(fldr)

                # NOTE: because multi-channel model has only 1 label, it has to be renamed to 2.0 to match
                # the label id with the region-based models
                if 'SCIsegV2_multi' in df['model'].values[0]:
                    df['label'] = 2.0

                df_sites = pd.concat([df_sites, df])

            df_folds = pd.concat([df_folds, df_sites])

        # print(df_folds)
        # compute the mean and std over all folds for each site
        # df_folds = df_folds.groupby(['label', 'site', 'model']).mean(numeric_only=True).reset_index()

        df_mega = pd.concat([df_mega, df_folds])
    
    # compute the mean and std over sites for each model
    df_mega = df_mega.groupby(['model', 'site', 'label']).mean(numeric_only=True).reset_index()
    # print(df_mega.reset_index(drop=True))

    # count the number of subjects per site
    for site in test_sites["SCI"]:
        csv_all = os.path.join(args.i[0], 'fold_0', f'test_{site}', f'{site}_metrics.csv')
        df = pd.read_csv(csv_all)
        # keep only the rows with label 2.0 (lesions)
        df = df[df['label'] == 2.0]
        num_subs_per_site[sites_to_rename[site]] = df.shape[0]

    # define the order for models
    order_models = [
        'AcuteSCI', 
        'DCM', 'SCIsegV1',
        'SCIsegV2_single\ndefaultDA', 'SCIsegV2_single\naggressiveDA',
        'SCIsegV2_multi\ndefaultDA', 'SCIsegV2_multi\naggressiveDA']
    
    print("Generating plots for Lesions")
    for metric in ['dsc', 'lesion_ppv', 'lesion_sensitivity', 'lesion_f1_score']:
        # keep the only the dataset, model, and metric columns
        df_metric = df_mega.copy()
        df_metric = df_metric[df_metric['label'] == 2.0]
        df_metric = df_metric[['model', 'site', f'{metric}_mean', f'{metric}_std']]

        # convert the metric values to float
        df_metric[f'{metric}_mean'] = df_metric[f'{metric}_mean'].astype(float)
        df_metric[f'{metric}_std'] = df_metric[f'{metric}_std'].astype(float)

        # use seaborn catplot to plot the metrics
        sns.set_theme(style="whitegrid")
        # # set font to Helvetica
        # plt.rcParams['font.family'] = 'serif'
        # plt.rcParams['font.serif'] = ['Helvetica'] #+ plt.rcParams['font.serif']
        # figure size
        g = sns.catplot(
            data=df_metric, x='site', y=f'{metric}_mean',
            hue='model', kind='bar', aspect=2, alpha=0.6, height=6,
            hue_order=order_models,
        )
        g.ax.figure.set_size_inches(14, 6.5)
        # y-axis limits
        g.set(ylim=(0, 1))
        g.ax.set_yticks(np.arange(0, 1.1, 0.1))
        # remove the legend
        g._legend.remove()       

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
        # set in bold and increase the font size
        if metric == 'dsc':
            g.set_axis_labels("", "Dice Score", fontsize=14, fontweight='bold')
        else:
            g.set_axis_labels("", f"{metric.upper()}", fontsize=14, fontweight='bold')

        # Add space on the bottom for the legend
        plt.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.275)
        
        # create a horizontal legend with the model names
        g.ax.legend(loc='upper center', ncol=num_models_to_compare, bbox_to_anchor=(0.5, -0.2),
                    fontsize=12.5)

        # Update the x-axis labels with the number of subjects per site
        new_labels = [f"{site}\n(n={num_subs_per_site[site]})" for site in df_metric['site'].unique()]
        g.ax.set_xticklabels(new_labels, fontsize=12, fontweight='bold')
            
        print(f"\tSaving the plot for {metric}")
        plt.savefig(os.path.join(path_out, f"{metric}_lesion.png"), dpi=300)

    # create a df only of lesions
    df_mega = df_mega[df_mega['label'] == 2.0]
    # keep only the SCIsegV2 and SCIsegV1 models
    df_mega = df_mega[df_mega['model'].str.contains('SCIsegV2|SCIsegV1')]
    # keep only the following metrics
    df_mega = df_mega[[
        'model', 'site', 
        'nsd_mean', 'nsd_std',
        'rel_vol_error_mean', 'rel_vol_error_std',
        'lesion_ppv_mean', 'lesion_ppv_std', 
        'lesion_sensitivity_mean', 'lesion_sensitivity_std', 
        'lesion_f1_score_mean', 'lesion_f1_score_std']]
    # print(df_mega.reset_index(drop=True)) #.to_latex(index=False))
    # save the df_mega dataframe to a csv file
    df_mega.to_csv(os.path.join(path_out, 'lesion_metrics.csv'), index=False)

    
    # print("Generating plots for SC")
    # for metric in metrics_sc:
    #     # keep the only the dataset, model, and metric columns
    #     df_metric = df_mega.copy()
    #     df_metric = df_metric[df_metric['label'] == 1.0]
    #     df_metric = df_metric[['model', 'site', f'{metric}_mean', f'{metric}_std']]

    #     # convert the metric values to float
    #     df_metric[f'{metric}_mean'] = df_metric[f'{metric}_mean'].astype(float)
    #     df_metric[f'{metric}_std'] = df_metric[f'{metric}_std'].astype(float)

    #     # use seaborn catplot to plot the metrics
    #     sns.set_theme(style="whitegrid")
    #     g = sns.catplot(
    #         data=df_metric, x='site', y=f'{metric}_mean',
    #         hue='model', kind='bar', aspect=2, alpha=0.6, height=6,
    #     )
    #     # y-axis limits
    #     g.set(ylim=(0, 1))
    #     g.ax.set_yticks(np.arange(0, 1.1, 0.1))

    #     # add error bars from the std values in hte dataframe
    #     for bar, std in zip(g.ax.patches, df_metric[f'{metric}_std']):
    #         bar_height = bar.get_height()
    #         # plot with dashed lines
    #         g.ax.errorbar(
    #             bar.get_x() + bar.get_width() / 2,
    #             bar_height, 
    #             yerr=std, fmt='o', markersize=3,
    #             color='k', linewidth=1,
    #             capsize=3, capthick=1,
    #         )

    #     g.despine(left=True)
    #     g.set_axis_labels("Site", f"{metric}")
    #     g.legend.set_title("Model")

    #     # Update the x-axis labels with the number of subjects per site
    #     new_labels = [f"{site}\n(n={num_subs_per_site[site]})" for site in df_metric['site'].unique()]
    #     g.ax.set_xticklabels(new_labels) #, rotation=45, ha='right')

    #     print(f"\tSaving the plot for {metric}_sc")
    #     plt.savefig(os.path.join(path_out, f"{metric}_sc.png"))


if __name__ == '__main__':
    main()