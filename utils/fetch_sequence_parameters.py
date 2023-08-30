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

import numpy as np
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

    # Read the JSON file, return dict with n/a if the file is empty
    try:
        with open(file_path) as f:
            data = json.load(f)
    except:
        print(f'WARNING: {file_path} is empty.')
        return {param: "n/a" for param in LIST_OF_PARAMETERS}

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
    print(f'Parsed data saved to {os.path.join(dir_path, "parsed_data.csv")}')

    # Print the min and max values of the MagneticFieldStrength, PixelSpacing, and SliceThickness
    print(df[['MagneticFieldStrength', 'PixelSpacing', 'SliceThickness']].agg([np.min, np.max]))

    # Print unique values of the Manufacturer and ManufacturerModelName
    print(df[['Manufacturer', 'ManufacturerModelName']].drop_duplicates())
    # Print number of filenames for unique values of the Manufacturer
    print(df.groupby('Manufacturer')['filename'].nunique())
    # Print number of filenames for unique values of the MagneticFieldStrength
    print(df.groupby('MagneticFieldStrength')['filename'].nunique())


if __name__ == '__main__':
    main()
