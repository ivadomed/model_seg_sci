"""
The script:
 - fetch subjects from the provided folders (each one containing the predictions for a specific seed)
 - keep only the unique subjects (to avoid duplicates between seeds)
 - construct paths to spinal cord and lesion predictions obtained using our 3D nnUNet model
 - run sct_analyze_lesion to compute the MRI-derived metrics

Authors: Jan Valosek, Naga Karthik

Example:
    python compute_lesion_metrics.py
    -i sci-multisite-inference_sc_seed123_2023-09-08/data_processed sci-multisite-inference_sc_seed42_2023-09-06/data_processed ...
"""

import os
import glob
import argparse

import pandas as pd


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='TODO',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-i',
        required=True,
        nargs='+',
        help='Space separated list of paths to the \'data_processed\' folders containing the spinal cord predictions. '
             'Note: The lesion predictions are fetched automatically from the corresponding \'data_processed\' folder. '
             'Example: sci-multisite-inference_sc_seed123_2023-09-08/data_processed '
             'sci-multisite-inference_sc_seed42_2023-09-06/data_processed'
    )
    parser.add_argument(
        '-o',
        required=False,
        default='stats',
        help='Path to the output folder where figures will be saved. Default: ./stats'
    )

    return parser


def get_fnames(dir_paths):
    """
    Get the list of file names across all input paths (i.e. all seeds)
    :param dir_paths: list of paths to the 'data_processed' folders containing the spinal cord predictions for each seed
    :return: pandas dataframe with the lesion and spinal cord file names
    """
    # Initialize an empty list file names across all input paths (i.e. all seeds)
    fname_files_all = list()

    # Get all file names across all input paths (i.e. all seeds)
    for dir_path in dir_paths:
        # Get SC segmentation produced by our 3D nnUNet model (_seg_nnunet_3d.nii.gz)
        fname_files = glob.glob(os.path.join(dir_path, '**', '*_seg_nnunet_3d.nii.gz'), recursive=True)
        # if fname_files is empty, exit
        if len(fname_files) == 0:
            print(f'ERROR: No _seg_nnunet_3d.nii.gz files found in {dir_path}')

        fname_files_all.extend(fname_files)

    # Sort the list of file names (to make the list the same when provided the input folders in different order)
    fname_files_all.sort()

    # Convert fname_files_all into pandas dataframe
    df = pd.DataFrame(fname_files_all, columns=['fname_sc'])

    # Add a column with participant_id
    df['participant_id'] = df['fname_sc'].apply(lambda x: os.path.basename(x))
    print(f'Number of rows: {len(df)}')

    # Keep only unique participant_id rows
    df = df.drop_duplicates(subset=['participant_id'])
    print(f'Number of unique participants: {len(df)}')

    # Add a column with fname_lesion by replacing _sc_ by _lesion_ and _seg_nnunet_3d with _lesion_nnunet_3d
    df['fname_lesion'] = df['fname_sc'].apply(lambda x: x.replace('_sc_', '_lesion_').
                                              replace('_seg_nnunet_3d', '_lesion_nnunet_3d'))

    # Make sure the lesion exists
    for fname in df['fname_lesion']:
        if not os.path.exists(fname):
            raise ValueError(f'ERROR: {fname} does not exist.')

    return df


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Output directory
    output_dir = os.path.join(os.getcwd(), args.o)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parse input paths
    dir_paths = [os.path.join(os.getcwd(), path) for path in args.i]

    # Check if the input path exists
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            raise ValueError(f'ERROR: {dir_path} does not exist.')

    # Get the lesion and spinal cord file names
    df = get_fnames(dir_paths)

    # Run sct_analyze_lesion to compute the lesion metrics:
    # - length
    # - volume
    # - maximum axial damage ratio
    for row in df.itertuples():
        print(f'Processing {row.fname_lesion}')
        os.system(f'sct_analyze_lesion -m {row.fname_lesion} -s {row.fname_sc} -o {output_dir}')

    print('here')


if __name__ == '__main__':
    main()
