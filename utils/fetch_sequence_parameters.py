"""
Loop across JSON sidecar files and nii headers in the input path and parse from them the following information:
    MagneticFieldStrength
    Manufacturer
    ManufacturerModelName
    ProtocolName
    PixDim
    SliceThickness

If JSON sidecar is not available (sci-paris), fetch only PixDim and SliceThickness from nii header.

Example usage:
    python utils/fetch_sequence_parameters.py -i /path/to/dataset -contrast T2w

Author: Jan Valosek
"""

import os
import glob
import json
import argparse

import numpy as np
import pandas as pd
import nibabel as nib

LIST_OF_PARAMETERS = [
    'MagneticFieldStrength',
    'Manufacturer',
    'ManufacturerModelName',
    'ProtocolName'
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
        help='Path to BIDS dataset. For example: sci-zurich or sci-colorado'
    )
    parser.add_argument(
        '-contrast',
        required=True,
        type=str,
        help='Image contrast. Examples: T2w (sci-colorado, sci-paris), acq-sag_T2w or acq-ax_T2w (sci-zurich)',
    )

    return parser


def parse_json_file(file_path):
    """
    Read the JSON file and parse from it relevant information.
    :param file_path:
    :return:
    """

    file_path = file_path.replace('.nii.gz', '.json')

    # Read the JSON file, return dict with n/a if the file is empty
    try:
        with open(file_path) as f:
            data = json.load(f)
    except:
        print(f'WARNING: {file_path} is empty.')
        return {param: "n/a" for param in LIST_OF_PARAMETERS}

    # Initialize an empty dictionary to store the parsed information
    parsed_info = {}

    if 'zurich' in file_path:
        # For sci-zurich, JSON file contains a list of dictionaries, each dictionary contains a list of dictionaries
        data = data['acqpar'][0]
    elif 'colorado' in file_path:
        data = data

    # Loop across the parameters
    for param in LIST_OF_PARAMETERS:
        try:
            parsed_info[param] = data[param]
        except:
            parsed_info[param] = "n/a"

    return parsed_info


def parse_nii_file(file_path):
    """
    Read nii file header using nibabel and to get PixDim and SliceThickness.
    We are doing this because 'PixelSpacing' and 'SliceThickness' can be missing from the JSON file.
    :param file_path:
    :return:
    """

    # Read the nii file, return dict with n/a if the file is empty
    try:
        img = nib.load(file_path)
        header = img.header
    except:
        print(f'WARNING: {file_path} is empty. Did you run git-annex get .?')
        return {param: "n/a" for param in ['PixDim', 'SliceThickness']}

    # Initialize an empty dictionary to store the parsed information
    parsed_info = {
        'PixDim': list(header['pixdim'][1:3]),
        'SliceThickness': float(header['pixdim'][3])
    }

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

    list_of_files = glob.glob(os.path.join(dir_path, '**', '*' + contrast + '.nii.gz'), recursive=True)
    # Keep only ses-01 for sci-zurich
    if 'zurich' in dir_path:
        list_of_files = [file for file in list_of_files if 'ses-01' in file]

    # Loop across JSON sidecar files in the input path
    for file in list_of_files:
        if file.endswith('.nii.gz'):
            print(f'Parsing {file}')
            file_path = os.path.join(dir_path, file)
            parsed_json = parse_json_file(file_path)
            parsed_header = parse_nii_file(file_path)
            # Note: **metrics is used to unpack the key-value pairs from the metrics dictionary
            parsed_data.append({'filename': file, **parsed_json, **parsed_header})

    # Create a pandas DataFrame from the parsed data
    df = pd.DataFrame(parsed_data)

    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(dir_path, 'parsed_data.csv'), index=False)
    print(f'Parsed data saved to {os.path.join(dir_path, "parsed_data.csv")}')

    # For sci-paris, we do not have JSON sidecars --> we can fetch only PixDim and SliceThickness from nii header
    if 'sci-paris' in dir_path:
        # Print the min and max values of the PixDim, and SliceThickness
        print(df[['PixDim', 'SliceThickness']].agg([np.min, np.max]))
    else:
        # Remove rows with n/a values for MagneticFieldStrength
        df = df[df['MagneticFieldStrength'] != 'n/a']

        # Convert MagneticFieldStrength to float
        df['MagneticFieldStrength'] = df['MagneticFieldStrength'].astype(float)

        # Print the min and max values of the MagneticFieldStrength, PixDim, and SliceThickness
        print(df[['MagneticFieldStrength', 'PixDim', 'SliceThickness']].agg([np.min, np.max]))

        # Print unique values of the Manufacturer and ManufacturerModelName
        print(df[['Manufacturer', 'ManufacturerModelName']].drop_duplicates())
        # Print number of filenames for unique values of the Manufacturer
        print(df.groupby('Manufacturer')['filename'].nunique())
        # Print number of filenames for unique values of the MagneticFieldStrength
        print(df.groupby('MagneticFieldStrength')['filename'].nunique())


if __name__ == '__main__':
    main()
