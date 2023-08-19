#
# Parse the xml files with segmentation metrics and create a rainplot
#
# Authors: Jan Valosek
#

import os
import glob
import argparse

import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import ptitprince as pt


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='Parse the xml files with segmentation metrics and create a rainplot',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-i',
        metavar="<folder>",
        required=True,
        type=str,
        help='Path to the folder containing the xml files.'
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


def fetch_filename_and_method(input_string):
    """
    Fetch the file and method from the input string
    :param input_string: input string, e.g. 'sub-5416_T2w_seg_nnunet'
    :return file: file name, e.g. 'sub-5416_T2w'
    :return method: segmentation method, e.g. 'nnunet'
    """
    file = input_string.split('_seg_')[0]
    method = input_string.split('_seg_')[1]

    return file, method


def create_rainplot(df, path_figures):
    """
    Create raincloud plots (violionplot + boxplot + individual points)
    :param df:
    :param path_figures:
    :return:
    """
    for metric in ['Jaccard', 'Dice', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'RelativeVolumeError',
                   'HausdorffDistance', 'ContourMeanDistance', 'SurfaceDistance']:
        ax = pt.RainCloud(data=df,
                          x='method',
                          y=metric,
                          order=['propseg', 'deepseg_2d', 'deepseg_3d', 'nnunet'],
                          linewidth=0,      # violionplot border line (0 - no line)
                          width_viol=.5,    # violionplot width
                          width_box=.3,     # boxplot width
                          rain_alpha=.7,    # individual points transparency - https://github.com/pog87/PtitPrince/blob/23debd9b70fca94724a06e72e049721426235f50/ptitprince/PtitPrince.py#L707
                          alpha=.7,         # violin plot transparency
                          box_showmeans=True,  # show mean value inside the boxplots
                          box_meanprops={'marker': '^', 'markerfacecolor': 'black', 'markeredgecolor': 'black',
                                         'markersize': '6'}
                          )
        # Move grid to background (i.e. behind other elements)
        ax.set_axisbelow(True)
        # Add horizontal grid lines
        ax.yaxis.grid(True)
        plt.tight_layout()

        # save figure
        fname_fig = os.path.join(path_figures, f'rainplot_{metric}.png')
        plt.savefig(fname_fig, dpi=300)
        plt.close()
        print(f'Created: {fname_fig}\n')


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    dir_path = os.path.abspath(args.i)

    if not os.path.isdir(dir_path):
        print(f'ERROR: {args.i} does not exist.')

    # Get all the xml files in the directory
    xml_files = glob.glob(os.path.join(dir_path, '*.xml'))
    # if xml_files is empty, exit
    if len(xml_files) == 0:
        print(f'ERROR: No xml files found in {dir_path}')

    # Initialize an empty list to store the parsed data
    parsed_data = []

    # Loop across xml files, parse them and aggregate the results into pandas dataframe
    for xml_file in xml_files:
        filename, segmentation_metrics = parse_xml_file(xml_file)
        # Note: **metrics is used to unpack the key-value pairs from the metrics dictionary
        parsed_data.append({'filename': filename, **segmentation_metrics})

    # Create a pandas DataFrame from the parsed data
    df = pd.DataFrame(parsed_data)

    # Apply the fetch_filename_and_method function to each row using a lambda function
    df[['file', 'method']] = df['filename'].apply(lambda x: pd.Series(fetch_filename_and_method(x)))
    # Reorder the columns
    df = df[['filename', 'file', 'method'] + [col for col in df.columns if col not in ['filename', 'file', 'method']]]

    # Make figure directory in dir_path
    path_figures = os.path.join(dir_path, 'figures')
    if not os.path.exists(path_figures):
        os.makedirs(path_figures)

    create_rainplot(df, path_figures)


if __name__ == '__main__':
    main()
