"""
Loop across JSON sidecar files in the input path and parse from them the following information:
    MagneticFieldStrength
    Manufacturer
    ManufacturerModelName
    ProtocolName
    PixelSpacing
    SliceThickness

Author: Jan Valosek
"""

import os
import glob
import json
import argparse

import pandas as pd

LIST_OF_PARAMETERS = [
    'MagneticFieldStrength',
    'Manufacturer',
    'ManufacturerModelName',
    'ProtocolName',
    'PixelSpacing',
    'SliceThickness',
]


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='Loop across JSON sidecar files in the input path and parse from them relevant information.',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-i',
        required=True,
        type=str,
        help='Path to BIDS dataset. For example: sci-zurich'
    )
    parser.add_argument(
        '-contrast',
        type=str,
        help='Image contrast. For example: acq-sag_T2w',
        default='acq-sag_T2w'
    )

    return parser


def parse_json_file(file_path):
    """
    Read the JSON file and parse from it relevant information.
    :param file_path:
    :return:
    """

    # Read the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Initialize an empty dictionary to store the parsed information
    parsed_info = {}

    # Loop across the parameters
    for param in LIST_OF_PARAMETERS:
        try:
            parsed_info[param] = data['acqpar'][0][param]
        except:
            parsed_info[param] = "n/a"

    return parsed_info


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    dir_path = os.path.abspath(args.i)

    if not os.path.isdir(dir_path):
        print(f'ERROR: {args.i} does not exist.')

    contrast = args.contrast

    # Initialize an empty list to store the parsed data
    parsed_data = []

    # Loop across JSON sidecar files in the input path
    for file in sorted(glob.glob(os.path.join(dir_path, '**', '*' + contrast + '.json'), recursive=True)):
        if file.endswith('.json'):
            print(f'Parsing {file} ...')
            file_path = os.path.join(dir_path, file)
            parsed_info = parse_json_file(file_path)
            # Note: **metrics is used to unpack the key-value pairs from the metrics dictionary
            parsed_data.append({'filename': file, **parsed_info})

    # Create a pandas DataFrame from the parsed data
    df = pd.DataFrame(parsed_data)

    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(dir_path, 'parsed_data.csv'), index=False)


if __name__ == '__main__':
    main()
