"""
The script:
 - fetch subjects from the provided /results folders (each one for a specific seed)
 - read XML files with ANIMA segmentation metrics (Dice, Surface Distance, ...) and execution_time.csv
 (located under /results) computed using sct_analyze_lesion separately for GT and predicted using our 3D nnUNet model)
 - average the metrics across seeds based (because some subjects are present in multiple seeds)
 - print the metrics into terminal for each method and each site (and also all sites together) in order to create
 Table 2 and Table 3
 - create Raincloud plot for each segmentation metrics (Dice, RVE, ...)
 - Raincloud plot are saved in the folder defined by the '-o' flag (Default: ./figures).

Authors: Jan Valosek, Naga Karthik

Example:
    python generate_figures.py
    -i sci-multisite-test-data_seed42_sc_inference_2023-09-11/results sci-multisite-test-data_seed123_sc_inference_2023-09-11/results ...
    -pred-type sc

    python generate_figures.py
    -i sci-multisite-test-data_seed42_lesion_inference_2023-09-11/results sci-multisite-test-data_seed123_lesion_inference_2023-09-11/results ...
    -pred-type lesion
"""

import os
import sys
import re
import glob
import argparse
import logging
import numpy as np
import matplotlib as mpl

import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import ptitprince as pt

from functools import reduce
from scipy.stats import wilcoxon, normaltest, kruskal
from statsmodels.stats.multitest import multipletests

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)


METHODS_TO_LABEL_SC = {
    'propseg': 'sct_propseg',
    'deepseg_2d': 'sct_deepseg_sc\n2D',
    'deepseg_3d': 'sct_deepseg_sc\n3D',
    'monai': 'contrast-agnostic',
    'nnunet_2d': 'SCIseg\n2D',
    'nnunet_3d': 'SCIseg\n3D'
    }

METHODS_TO_LABEL_LESION = {
    'nnunet_2d': 'SCIseg 2D',
    'nnunet_3d': 'SCIseg 3D',
    }

LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12
PALETTE = ['red', 'darkblue']


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


def format_pvalue(p_value, decimal_places=3, include_space=False, include_equal=True):
    """
    Format p-value.
    If the p-value is lower than alpha, format it to "<0.001", otherwise, round it to three decimals

    :param p_value: input p-value as a float
    :param decimal_places: number of decimal places the p-value will be rounded
    :param include_space: include space or not (e.g., ' = 0.06')
    :param include_equal: include equal sign ('=') to the p-value (e.g., '=0.06') or not (e.g., '0.06')
    :return: p_value: the formatted p-value (e.g., '<0.05') as a str
    """
    if include_space:
        space = ' '
    else:
        space = ''

    # If the p-value is lower than alpha, return '<alpha' (e.g., <0.001)
    for alpha in [0.001, 0.01, 0.05]:
        if p_value < alpha:
            p_value = space + "<" + space + str(alpha)
            break
    # If the p-value is greater than 0.05, round it number of decimals specified by decimal_places
    else:
        if include_equal:
            p_value = space + '=' + space + str(round(p_value, decimal_places))
        else:
            p_value = space + str(round(p_value, decimal_places))

    return p_value


def parse_xml_file(file_path):
    """
    Fetch filename and segmentation metrics from the xml file:

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
    :return filename: filename, e.g. 'sub-5416_T2w_seg_deepseg_2d_global'
    :return segmentation_metrics: dictionary with segmentation metrics
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


def fetch_participant_id_site_and_method(input_string, pred_type):
    """
    Fetch the participant_id, site and method from the input string
    :param input_string: input string, e.g. 'sub-5416_T2w_seg_nnunet'
    :return participant_id: subject id, e.g. 'sub-5416'
    :return session_id: session id, e.g. 'ses-01'
    :return site: site name, e.g. 'zurich' or 'colorado'
    :return method: segmentation method, e.g. 'nnunet'
    """

    # Fetch participant_id
    participant = re.search('sub-(.*?)[_/]', input_string)  # [_/] slash or underscore
    participant_id = participant.group(0)[:-1] if participant else ""  # [:-1] removes the last underscore or slash

    # Fetch session_id
    session = re.search('ses-(.*?)[_/]', input_string)  # [_/] slash or underscore
    session_id = session.group(0)[:-1] if session else ""  # [:-1] removes the last underscore or slash

    # Fetch site
    if 'sub-zh' in input_string:
        site = 'zurich'
    elif re.match(r'sub-\d{4}', input_string):
        site = 'colorado'
    elif re.match(r'sub-\d{2}', input_string):
        site = 'dcm-zurich'

    # Fetch method
    if pred_type == 'sc':
        if site == 'dcm-zurich':
            method = input_string.split('_sc_')[1]
        else:
            method = input_string.split('_seg_')[1]
    elif pred_type == 'lesion':
        method = input_string.split('_lesion_')[1]
    else:
        raise ValueError(f'Unknown pred_type: {pred_type}')

    return participant_id, session_id, site, method


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
        logger.info(f'{metric}:')
        # Loop across methods (e.g., nnUNet 2D, nnUNet 3D, etc.)
        for method in df['method'].unique():
            num_of_subjects = len(df['participant_id'].unique())
            # Mean +- std across sites
            if pred_type == 'sc':
                logger.info(f'\t{method} (all sites, n={num_of_subjects}): '
                            f'{df[df["method"] == method][metric].mean():.2f} +/- '
                            f'{df[df["method"] == method][metric].std():.2f}')
            elif pred_type == 'lesion':
                logger.info(f'\t{method} (all sites, n={num_of_subjects}): '
                            f'{df[df["method"] == method][metric].mean():.2f} +/- '
                            f'{df[df["method"] == method][metric].std():.2f}')
            # Loop across sites
            for site in df['site'].unique():
                df_tmp = df[(df['method'] == method) & (df['site'] == site)]
                num_of_subjects = len(df_tmp['participant_id'].unique())
                if pred_type == 'sc':
                    logger.info(f'\t{method} ({site}, n={num_of_subjects}): '
                                f'{df_tmp[metric].mean():.2f} ± {df_tmp[metric].std():.2f}')
                elif pred_type == 'lesion':
                    logger.info(f'\t{method} ({site}, n={num_of_subjects}): '
                                f'{df_tmp[metric].mean():.2f} ± {df_tmp[metric].std():.2f}')


def split_string_by_capital_letters(s):
    """
    Split a string by capital letters
    :param s: e.g., 'RelativeVolumeError'
    :return: e.g., 'Relative Volume Error'
    """
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', s)


def create_rainplot(df, list_of_metrics, path_figures, pred_type, num_of_seeds):
    """
    Create Raincloud plots (violionplot + boxplot + individual points)
    :param df: dataframe with segmentation metrics
    :param list_of_metrics: list of metrics to be plotted
    :param path_figures: path to the folder where the figures will be saved
    :param pred_type: type of prediction to create plots for; sc: spinal cord segmentation; lesion: lesion segmentation
    :param num_of_seeds: number of seeds (obtained from the number of input folders)
    :return:
    """

    mpl.rcParams['font.family'] = 'Helvetica'

    # Capitalize site names from 'zurich' to 'Zurich' and from 'colorado' to 'Colorado' (to have nice legend)
    df['site'] = df['site'].apply(lambda x: x.capitalize())

    # Rename site names: "Zurich" -> "Site 1" and "Colorado" -> "Site 2"
    df['site'] = df['site'].apply(lambda x: 'Site 1' if x == 'Zurich' else 'Site 2')

    for metric in list_of_metrics:

        # Drop rows with NaN values for the current metric
        df_temp = df.dropna(subset=[metric])

        fig_size = (10, 5) if pred_type == 'sc' else (8, 5)
        fig, ax = plt.subplots(figsize=fig_size)
        ax = pt.RainCloud(data=df_temp,
                          x='method',
                          y=metric,
                          hue='site',
                          palette=PALETTE,
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
                          hue_order=['Site 1', 'Site 2'],
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
            n = len(df_temp[(df_temp['site'] == label) & (df_temp['method'] == 'nnunet_3d')]['participant_id'])
            labels[i] = f'{label} ' + '($\it{n}$' + f' = {n})'
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
            ax.set_ylim(-125, 125)

        if metric == 'SurfaceDistance' and pred_type == 'sc':
            ax.set_ylim(0, 15)

        # Set title
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
        logger.info(f'Created: {fname_fig}')


def print_colorado_subjects_with_dice_0(df_concat):
    """
    Print subjects with Dice = 0 for Colorado site
    :param df_concat: dataframe containing all the data
    :return:
    """
    df = df_concat[df_concat['site'] == 'colorado']
    df = df[df['Dice'] == 0]
    logger.info(f'Subjects with Dice = 0 for Colorado site:')
    logger.info(df[['participant_id', 'method', 'Dice']])


def compute_wilcoxon_test(df_concat, list_of_metrics):
    """
    Compute Wilcoxon signed-rank test (two related paired samples -- a same subject for nnunet_3d vs nnunet_2d)
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
    :param df_concat: dataframe containing all the data
    :param list_of_metrics: list of metrics to compute the Wilcoxon test for
    :return:
    """

    logger.info('')

    # Remove 'NbTestedLesions' and 'VolTestedLesions' from the list of metrics
    list_of_metrics = [metric for metric in list_of_metrics if metric not in ['NbTestedLesions', 'VolTestedLesions']]

    # Loop across sites
    for site in df_concat['site'].unique():
        # Loop across metrics
        for metric in list_of_metrics:
            # Reorder the dataframe
            df_nnunet_2d = df_concat[(df_concat['method'] == 'nnunet_2d') & (df_concat['site'] == site)]
            df_nnunet_3d = df_concat[(df_concat['method'] == 'nnunet_3d') & (df_concat['site'] == site)]

            # Combine the two dataframes based on participant_id and seed. Keep only metric column
            df = pd.merge(df_nnunet_2d[['participant_id', 'session_id', metric]],
                          df_nnunet_3d[['participant_id', 'session_id', metric]],
                          on=['participant_id', 'session_id'],
                          suffixes=('_2d', '_3d'))

            # Drop rows with NaN values
            df = df.dropna()
            # Print number of subjects
            logger.info(f'{metric}, {site}: Number of subjects: {len(df)}')

            # Run normality test
            stat, p = normaltest(df[metric + '_2d'])
            logger.info(f'{metric}, {site}: Normality test for nnunet_2d: formatted p{format_pvalue(p)}, '
                        f'unformatted p={p:0.6f}')
            stat, p = normaltest(df[metric + '_3d'])
            logger.info(f'{metric}, {site}: Normality test for nnunet_3d: formatted p{format_pvalue(p)}, '
                        f'unformatted p={p:0.6f}')

            # Compute Wilcoxon signed-rank test
            stat, p = wilcoxon(df[metric + '_2d'], df[metric + '_3d'])
            logger.info(f'{metric}, {site}: Wilcoxon signed-rank test between nnunet_2d and nnunet_3d: '
                        f'formatted p{format_pvalue(p)}, unformatted p={p:0.6f}')


def compute_kruskal_wallis_test(df_concat, list_of_metrics):
    """
    Compute Kruskal-Wallis H-test (non-parametric version of ANOVA)
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
    :param df_concat:
    :param list_of_metrics:
    :return:
    """

    logger.info('')

    # Remove 'NbTestedLesions' and 'VolTestedLesions' from the list of metrics
    list_of_metrics = [metric for metric in list_of_metrics if metric not in ['NbTestedLesions', 'VolTestedLesions']]

    # Loop across sites
    for site in df_concat['site'].unique():
        # Loop across metrics
        for metric in list_of_metrics:
            # Reorder the dataframe
            df_propseg = df_concat[(df_concat['method'] == 'propseg') & (df_concat['site'] == site)]
            df_deepseg_2d = df_concat[(df_concat['method'] == 'deepseg_2d') & (df_concat['site'] == site)]
            df_deepseg_3d = df_concat[(df_concat['method'] == 'deepseg_3d') & (df_concat['site'] == site)]
            df_contrast_agnostic = df_concat[(df_concat['method'] == 'monai') & (df_concat['site'] == site)]
            df_nnunet_2d = df_concat[(df_concat['method'] == 'nnunet_2d') & (df_concat['site'] == site)]
            df_nnunet_3d = df_concat[(df_concat['method'] == 'nnunet_3d') & (df_concat['site'] == site)]

            # Combine all dataframes based on participant_id and seed. Keep only metric column. Use reduce and lambda
            df = reduce(lambda left, right: pd.merge(left, right, on=['participant_id', 'session_id']),
                        [df_propseg[['participant_id', 'session_id', metric]],
                         df_deepseg_2d[['participant_id', 'session_id', metric]],
                         df_deepseg_3d[['participant_id', 'session_id', metric]],
                         df_contrast_agnostic[['participant_id', 'session_id', metric]],
                         df_nnunet_2d[['participant_id', 'session_id', metric]],
                         df_nnunet_3d[['participant_id', 'session_id', metric]]])
            # Rename columns
            df.columns = ['participant_id', 'session_id', metric + '_propseg', metric + '_deepseg_2d',
                            metric + '_deepseg_3d', metric + '_monai', metric + '_nnunet_2d', metric + '_nnunet_3d']

            # Drop rows with NaN values
            df = df.dropna()
            # Print number of subjects
            logger.info(f'{metric}, {site}: Number of subjects: {len(df)}')

            # Compute Kruskal-Wallis H-test
            stat, p = kruskal(df[metric + '_propseg'], df[metric + '_deepseg_2d'], df[metric + '_deepseg_3d'],
                              df[metric + '_monai'], df[metric + '_nnunet_2d'], df[metric + '_nnunet_3d'])
            logger.info(f'{metric}, {site}: Kruskal-Wallis H-test: formatted p{format_pvalue(p)}, '
                        f'unformatted p={p:0.6f}')

            # Run post-hoc tests between nnunet_3d and all other methods
            if p < 0.05:
                p_val_dict = {}
                for method in ['propseg', 'deepseg_2d', 'deepseg_3d', 'monai', 'nnunet_2d']:
                    stats, p = wilcoxon(df[metric + '_nnunet_3d'], df[metric + '_' + method], alternative='two-sided')
                    p_val_dict[method] = p
                # Do Bonferroni correction
                # https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
                _, pvals_corrected, _, _ = multipletests(list(p_val_dict.values()), alpha=0.05, method='bonferroni')
                p_val_dict_corrected = dict(zip(list(p_val_dict.keys()), pvals_corrected))
                # Format p-values using format_pvalue function
                p_val_dict_corrected = {k: format_pvalue(v) for k, v in p_val_dict_corrected.items()}
                logger.info(f'{metric}, {site}: Post-hoc tests between nnunet_3d and all other methods:\n'
                            f'{p_val_dict_corrected}')


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    pred_type = args.pred_type

    # Output directory where the figures will be saved
    output_dir = os.path.join(os.getcwd(), args.o)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Dump log file there
    fname_log = f'log_stats_{pred_type}.txt'
    if os.path.exists(fname_log):
        os.remove(fname_log)
    fh = logging.FileHandler(os.path.join(os.path.abspath(output_dir), fname_log))
    logging.root.addHandler(fh)

    # Parse input paths
    dir_paths = [os.path.join(os.getcwd(), path) for path in args.i]

    # Check if the input path exists
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            raise ValueError(f'ERROR: {dir_path} does not exist.')

    # Initialize an empty list to store the parsed dataframes from each directory
    list_of_df = list()

    # Loop across provided directories and parse the xml files
    for dir_path in dir_paths:
        # Get all the xml files in the directory
        xml_files = glob.glob(os.path.join(dir_path, f'*_{pred_type}_*.xml'))
        # if xml_files is empty, exit
        if len(xml_files) == 0:
            print(f'ERROR: No xml files found in {dir_path}')
            sys.exit(1)

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

        logger.info(f'Parsed {len(df)} files for seed {seed} from {dir_path}.')

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
        df[['participant_id', 'session_id', 'site', 'method']] = df['filename'].\
            apply(lambda x: pd.Series(fetch_participant_id_site_and_method(x, pred_type)))
        # Reorder the columns
        df = df[['filename', 'participant_id', 'session_id', 'site', 'method'] + [col for col in df.columns if col not in ['filename', 'participant_id', 'session_id', 'site', 'method']]]

        # remove '_fullres' from the method column
        df['method'] = df['method'].apply(lambda x: x.replace('_fullres', ''))

        list_of_df.append(df)

    # Concatenate the list of dataframes into a single dataframe
    df_concat = pd.concat(list_of_df, ignore_index=True)

    # If a participant_id is duplicated (because the test image is presented across multiple seeds), average the
    # metrics across seeds for the same subject.
    df_concat = df_concat.groupby(['participant_id', 'session_id', 'site', 'method']).mean().reset_index()

    # Remove 'sub-5740' (https://github.com/ivadomed/model_seg_sci/issues/59)
    logger.info(f'Removing subject sub-5740 from the dataframe.')
    df_concat = df_concat[df_concat['participant_id'] != 'sub-5740']

    # Sort the dataframe by participant_id and seed
    df_concat = df_concat.sort_values(by=['participant_id'])

    # Print colorado subjects with Dice=0
    print_colorado_subjects_with_dice_0(df_concat)

    # For lesions, compute Wilcoxon signed-rank test test between nnunet_3d and nnunet_2d
    if pred_type == 'lesion':
        compute_wilcoxon_test(df_concat, list_of_metrics)
    # For SC, compute Kruskal-Wallis H-test (we have 6 methods)
    else:
        compute_kruskal_wallis_test(df_concat, list_of_metrics)

    # Print mean and std for each metric
    print_mean_and_std(df_concat, list_of_metrics, pred_type)

    logger.info("")
    # Create Raincloud plot for each metric
    num_of_seeds = len(dir_paths)
    create_rainplot(df_concat, list_of_metrics, output_dir, pred_type, num_of_seeds)


if __name__ == '__main__':
    main()