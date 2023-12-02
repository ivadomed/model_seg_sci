"""
Parse the xml files with segmentation metrics and execution_time.csv and create Raincloud plot.
Raincloud plot are saved in the folder defined by the '-o' flag (Default: ./figures).

Authors: Jan Valosek, Naga Karthik

Example:
    python generate_figures.py
    -i sci-multisite-test-data_seed42_sc_inference_2023-09-11/results sci-multisite-test-data_seed123_sc_inference_2023-09-11/results
    -pred-type sc
"""

import os
import re
import glob
import argparse
import numpy as np
import matplotlib as mpl

import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import ptitprince as pt


METHODS_TO_LABEL_SC = {
    'propseg': 'sct_propseg',
    'deepseg_2d': 'sct_deepseg_sc 2D',
    'deepseg_3d': 'sct_deepseg_sc 3D',
    'nnunet_2d': 'nnUNet 2D',
    'nnunet_3d': 'nnUNet 3D',
    'monai': 'contrast-agnostic',
    }

METHODS_TO_LABEL_LESION = {
    'nnunet_2d': 'nnUNet 2D',
    'nnunet_3d': 'nnUNet 3D',
    }

LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 12


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='Parse the xml files with segmentation metrics and execution_time.csv file and create Raincloud '
                    'plot. '
                    'Raincloud plot are saved in the folder defined by the \'-o\' flag (Default: ./figures).',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-i',
        required=True,
        nargs='+',
        help='Path to the folder(s) containing the xml files. You can provide multiple folders using a list separated '
             'by spaces.'
             'Example: sci-multisite-test-data_seed42_sc_inference_2023-09-11/results '
             'sci-multisite-test-data_seed123_sc_inference_2023-09-11/results'
    )
    parser.add_argument(
        '-pred-type',
        required=True,
        type=str,
        choices=['sc', 'lesion'],
        help='Type of prediction to create plots for. sc: spinal cord segmentation; lesion: lesion segmentation.'
    )
    parser.add_argument(
        '-o',
        required=False,
        default='figures',
        help='Path to the output folder where figures will be saved. Default: ./figures'
    )

    return parser


def parse_xml_file(file_path):
    """
    Fetch subject_id and segmentation metrics from the xml file:

    <?xml version="1.0" encoding="UTF-8"?>
    <image name="sub-5546_T2w_seg_deepseg_2d_global">
        <measure name="Jaccard">0.792123</measure>
        <measure name="Dice">0.884005</measure>
        <measure name="Sensitivity">0.843111</measure>
        <measure name="Specificity">0.999890</measure>
        <measure name="PPV">0.929069</measure>
        <measure name="NPV">0.999731</measure>
        <measure name="RelativeVolumeError">-9.252139</measure>
        <measure name="HausdorffDistance">5.400453</measure>
        <measure name="ContourMeanDistance">0.375997</measure>
        <measure name="SurfaceDistance">0.000526</measure>
    </image>

    :param file_path: path to the xml file
    :return: subject_id: subject id
    """
    # Parse the XML data
    with open(file_path, 'r') as xml_file:
        xml_data = xml_file.read()

    root = ET.fromstring(xml_data)

    filename = root.attrib['name']
    segmentation_metrics = {}

    for measure in root.findall('measure'):
        metric_name = measure.attrib['name']
        metric_value = float(measure.text)
        segmentation_metrics[metric_name] = metric_value

    # remove '_global' from the filename
    filename = filename.replace('_global', '')

    return filename, segmentation_metrics


def fetch_site_and_method(input_string, pred_type):
    """
    Fetch the file and method from the input string
    :param input_string: input string, e.g. 'sub-5416_T2w_seg_nnunet'
    :return site: site name, e.g. 'zurich' or 'colorado'
    :return method: segmentation method, e.g. 'nnunet'
    """
    if 'sub-zh' in input_string:
        site = 'zurich'
    else:
        site = 'colorado'
    
    if pred_type == 'sc':
        method = input_string.split('_seg_')[1]
    elif pred_type == 'lesion':
        method = input_string.split('_lesion_')[1]
    else:
        raise ValueError(f'Unknown pred_type: {pred_type}')

    return site, method


def print_mean_and_std(df, list_of_metrics, pred_type):
    """
    Print the mean and std for each metric
    :param df: dataframe with segmentation metrics
    :param list_of_metrics: list of metrics
    :param pred_type: type of prediction to create plots for; sc: spinal cord segmentation; lesion: lesion segmentation
    :return:
    """
    # Loop across metrics
    for metric in list_of_metrics:
        print(f'{metric}:')
        # Loop across methods (e.g., nnUNet 2D, nnUNet 3D, etc.)
        for method in df['method'].unique():
            # Mean +- std across sites
            if pred_type == 'sc':
                print(f'\t{method} (all sites): {df[df["method"] == method][metric].mean():.2f} +/- '
                      f'{df[df["method"] == method][metric].std():.2f}')
            elif pred_type == 'lesion':
                print(f'\t{method} (all sites): {df[df["method"] == method][metric].mean():.2f} +/- '
                      f'{df[df["method"] == method][metric].std():.2f}')
            # Loop across sites
            for site in df['site'].unique():
                df_tmp = df[(df['method'] == method) & (df['site'] == site)]
                if pred_type == 'sc':
                    print(f'\t{method} ({site}): {df_tmp[metric].mean():.2f} ± {df_tmp[metric].std():.2f}')
                elif pred_type == 'lesion':
                    print(f'\t{method} ({site}): {df_tmp[metric].mean():.2f} ± {df_tmp[metric].std():.2f}')


def split_string_by_capital_letters(s):
    """
    Split a string by capital letters
    :param s: e.g., 'RelativeVolumeError'
    :return: e.g., 'Relative Volume Error'
    """
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', s)


def create_rainplot(df, list_of_metrics, path_figures, pred_type):
    """
    Create Raincloud plots (violionplot + boxplot + individual points)
    :param df: dataframe with segmentation metrics
    :param list_of_metrics: list of metrics to be plotted
    :param path_figures: path to the folder where the figures will be saved
    :param pred_type: type of prediction to create plots for; sc: spinal cord segmentation; lesion: lesion segmentation
    :return:
    """

    mpl.rcParams['font.family'] = 'Arial'

    # Capitalize site names from 'zurich' to 'Zurich' and from 'colorado' to 'Colorado' (to have nice legend)
    df['site'] = df['site'].apply(lambda x: x.capitalize())

    for metric in list_of_metrics:
        fig_size = (10, 5) if pred_type == 'sc' else (8, 5)
        fig, ax = plt.subplots(figsize=fig_size)
        ax = pt.RainCloud(data=df,
                          x='method',
                          y=metric,
                          hue='site',
                          order=METHODS_TO_LABEL_SC.keys() if pred_type == 'sc' else METHODS_TO_LABEL_LESION.keys(),
                          dodge=True,       # move boxplots next to each other
                          linewidth=0,      # violionplot border line (0 - no line)
                          width_viol=.5,    # violionplot width
                          width_box=.3,     # boxplot width
                          rain_alpha=.7,    # individual points transparency - https://github.com/pog87/PtitPrince/blob/23debd9b70fca94724a06e72e049721426235f50/ptitprince/PtitPrince.py#L707
                          rain_s=2,         # individual points size
                          alpha=.7,         # violin plot transparency
                          box_showmeans=True,  # show mean value inside the boxplots
                          box_meanprops={'marker': '^', 'markerfacecolor': 'black', 'markeredgecolor': 'black',
                                         'markersize': '6'},
                          hue_order=['Zurich', 'Colorado'],
                          )

        # TODO: include mean +- std for each boxplot above the mean value

        # Change boxplot opacity (.0 means transparent)
        # https://github.com/mwaskom/seaborn/issues/979#issuecomment-1144615001
        for patch in ax.patches:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .0))

        # Include number of subjects for each site into the legend
        handles, labels = ax.get_legend_handles_labels()
        for i, label in enumerate(labels):
            n = len(df[(df['site'] == label) & (df['method'] == 'nnunet_3d')]['filename'])
            labels[i] = f'{label} (n={n})'
        # Since the figure contains violionplot + boxplot + scatterplot we are keeping only last two legend entries
        handles = handles[-2:]
        labels = labels[-2:]
        ax.legend(handles, labels, fontsize=TICK_FONT_SIZE)

        # Make legend box's frame color black and remove transparency
        legend = ax.get_legend()
        legend.legendPatch.set_facecolor('white')
        legend.legendPatch.set_edgecolor('black')

        # Remove spines
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(True)

        # Remove x-axis label
        ax.set_xlabel('')
        # Modify x-ticks labels
        ax.set_xticklabels(METHODS_TO_LABEL_SC.values() if pred_type == 'sc' else METHODS_TO_LABEL_LESION.values(),
                           fontsize=TICK_FONT_SIZE)
        # Increase y-axis label font size
        if metric == 'RelativeVolumeError':
            ax.set_ylabel(split_string_by_capital_letters(metric) + ' [%]', fontsize=TICK_FONT_SIZE)
        else:
            ax.set_ylabel(metric, fontsize=TICK_FONT_SIZE)
        # Increase y-ticks font size
        ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)

        # Adjust y-lim for 'RelativeVolumeError' metric
        if metric == 'RelativeVolumeError' and pred_type == 'sc':
            ax.set_ylim(-95, 62)
        elif metric == 'RelativeVolumeError' and pred_type == 'lesion':
            ax.set_ylim(-105, 105)

        # Set title
        num_of_seeds = len(df['seed'].unique())
        if pred_type == 'sc':
            ax.set_title(f'Test {split_string_by_capital_letters(metric)} for Spinal Cord Segmentation across {num_of_seeds} seeds',
                         fontsize=LABEL_FONT_SIZE)
        else:
            ax.set_title(f'Test {split_string_by_capital_letters(metric)} for Lesion Segmentation across {num_of_seeds} seeds',
                         fontsize=LABEL_FONT_SIZE)

        # Move grid to background (i.e. behind other elements)
        ax.set_axisbelow(True)
        # Add horizontal grid lines and change its opacity
        ax.yaxis.grid(True, alpha=0.3)
        # modify the y-axis ticks
        if metric == "Dice":
            ax.set_yticks(np.arange(0, 1.1, 0.1))

        plt.tight_layout()

        # save figure
        fname_fig = os.path.join(path_figures, f'{pred_type}_rainplot_{metric}.png')
        plt.savefig(fname_fig, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Created: {fname_fig}')


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    pred_type = args.pred_type

    # Output directory where the figures will be saved
    output_dir = os.path.join(os.getcwd(), args.o)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parse input paths
    dir_paths = [os.path.join(os.getcwd(), path) for path in args.i]

    # Check if the input path exists
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            raise ValueError(f'ERROR: {dir_path} does not exist.')

    # Initialize an empty list to store the parsed dataframes from each directory
    list_of_df = list()

    for dir_path in dir_paths:
        # Get all the xml files in the directory
        xml_files = glob.glob(os.path.join(dir_path, '*.xml'))
        # if xml_files is empty, exit
        if len(xml_files) == 0:
            print(f'ERROR: No xml files found in {dir_path}')

        # Fetch train-test split seed (e.g., 42) from the directory path
        seed = re.search(r'seed(\d+)', dir_path).group(1)

        # Initialize an empty list to store the parsed data
        parsed_data = []

        # Loop across xml files, parse them and aggregate the results into pandas dataframe
        for xml_file in xml_files:
            filename, segmentation_metrics = parse_xml_file(xml_file)
            # Note: **metrics is used to unpack the key-value pairs from the metrics dictionary
            parsed_data.append({'filename': filename, 'seed': seed, **segmentation_metrics})

        # Create a pandas DataFrame from the parsed data
        df = pd.DataFrame(parsed_data)

        print(f'Parsed {len(df)} files for seed {seed} from {dir_path}.')

        # Get list of ANIMA metrics
        list_of_metrics = list(df.columns)
        list_of_metrics.remove('filename')
        list_of_metrics.remove('seed')

        # Read execution_time.csv file and name first column as 'filename' and the second column as 'execution_time'
        fname_execution_time = os.path.join(dir_path, 'execution_time.csv')
        if os.path.isfile(fname_execution_time):
            df_execution_time = pd.read_csv(fname_execution_time, header=None, names=['filename', 'ExecutionTime[s]'])
            # Merge the two dataframes
            df = pd.merge(df, df_execution_time, on='filename')
            # Add execution time to the list of metrics
            list_of_metrics.append('ExecutionTime[s]')

        # Apply the fetch_filename_and_method function to each row using a lambda function
        df[['site', 'method']] = df['filename'].apply(lambda x: pd.Series(fetch_site_and_method(x, pred_type)))
        # Reorder the columns
        df = df[['filename', 'site', 'method'] + [col for col in df.columns if col not in ['filename', 'site', 'method']]]

        # remove '_fullres' from the method column
        df['method'] = df['method'].apply(lambda x: x.replace('_fullres', ''))

        list_of_df.append(df)

    # Concatenate the list of dataframes into a single dataframe
    df_concat = pd.concat(list_of_df, ignore_index=True)

    # Print mean and std for each metric
    print_mean_and_std(df_concat, list_of_metrics, pred_type)

    print("")
    # Create Raincloud plot for each metric
    create_rainplot(df_concat, list_of_metrics, output_dir, pred_type)


if __name__ == '__main__':
    main()
