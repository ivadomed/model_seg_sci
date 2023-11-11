"""
The script:
 - fetch subjects from the provided /results folders (each one for a specific seed)
 - keep only the unique subjects (to avoid duplicates between seeds)
 - read XLS files (located under /results) with lesion metrics (computed using sct_analyze_lesion separately for GT
 and predicted using our 3D nnUNet model)

Author: Jan Valosek
"""

import os
import re
import glob
import argparse
import subprocess

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
        help='Space separated list of paths to the \'results\' folders with XLS files. Each \'results\' folder '
             'corresponds to a specific seed. The results folders were generated using the \'01_analyze_lesions.sh\'.'
             'The XLS files contain lesion metrics and were generated using \'sct_analyze_lesion.\' '
             'Example: sci-multisite_analyze_lesions_seed7/results sci-multisite_analyze_lesions_seed42/results'
    )
    parser.add_argument(
        '-o',
        required=False,
        default='stats',
        help='Path to the output folder where stats and figures will be saved. Default: ./stats'
    )

    return parser


def fetch_subject_and_session(filename_path):
    """
    Get subject ID, session ID and filename from the input BIDS-compatible filename or file path
    The function works both on absolute file path as well as filename
    :param filename_path: input nifti filename (e.g., sub-001_ses-01_T1w.nii.gz) or file path
    (e.g., /home/user/MRI/bids/derivatives/labels/sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz
    :return: subject_session: subject ID and session ID (e.g., sub-001_ses-01) or subject ID (e.g., sub-001)
    """

    subject = re.search('sub-(.*?)[_/]', filename_path)     # [_/] means either underscore or slash
    subjectID = subject.group(0)[:-1] if subject else ""    # [:-1] removes the last underscore or slash

    session = re.search('ses-(.*?)[_/]', filename_path)     # [_/] means either underscore or slash
    sessionID = session.group(0)[:-1] if session else ""    # [:-1] removes the last underscore or slash
    # REGEX explanation
    # . - match any character (except newline)
    # *? - match the previous element as few times as possible (zero or more times)

    subject_session = subjectID + '_' + sessionID if subjectID and sessionID else subjectID

    return subject_session


def get_fnames(dir_paths):
    """
    Get the list of file names across all input paths (i.e. all seeds)
    :param dir_paths: list of paths to the 'results' folders containing the XLS files with lesion metrics for each seed
    :return: pandas dataframe with the paths to the XLS files for manual and predicted lesions
    """
    # Initialize an empty list file names across all input paths (i.e. all seeds)
    fname_files_all = list()

    # Get all file names across all input paths (i.e. all seeds)
    for dir_path in dir_paths:
        # Get lesion metrics for lesions predicted by our 3D SCIseg nnUNet model (_lesion_nnunet_3d_analysis.xls)
        fname_files = glob.glob(os.path.join(dir_path, '**', '*_lesion_nnunet_3d_analysis.xls'), recursive=True)
        # if fname_files is empty, exit
        if len(fname_files) == 0:
            print(f'ERROR: No _lesion_nnunet_3d_analysis.xls files found in {dir_path}')

        fname_files_all.extend(fname_files)

    # Sort the list of file names (to make the list the same when provided the input folders in different order)
    fname_files_all.sort()

    # Convert fname_files_all into pandas dataframe
    df = pd.DataFrame(fname_files_all, columns=['fname_lesion_nnunet_3d'])

    # Add a column with participant_id
    df['participant_id'] = df['fname_lesion_nnunet_3d'].apply(lambda x: fetch_subject_and_session(x))
    # Make the participant_id column the first column
    df = df[['participant_id', 'fname_lesion_nnunet_3d']]
    print(f'Number of rows: {len(df)}')

    # Keep only unique participant_id rows
    df = df.drop_duplicates(subset=['participant_id'])
    print(f'Number of unique participants: {len(df)}')

    # Add a column with fname_lesion_manual by replacing '_nnunet_3d' by '-manual_bin'
    df['fname_lesion_manual'] = df['fname_lesion_nnunet_3d'].apply(lambda x: x.replace('_nnunet_3d', '-manual_bin'))

    # Make sure the lesion exists
    for fname in df['fname_lesion_manual']:
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
        print(f'Created {output_dir}')

    # Parse input paths
    dir_paths = [os.path.join(os.getcwd(), path) for path in args.i]

    # Check if the input path exists
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            raise ValueError(f'ERROR: {dir_path} does not exist.')

    # For each participant_id, get the lesion and spinal cord file names
    df = get_fnames(dir_paths)

    # Iterate over the rows of the dataframe and read the XLS files
    for index, row in df.iterrows():

        print(f'Processing XLS files for {row["participant_id"]}')

        # Read the XLS file with lesion metrics for lesion predicted by our 3D SCIseg nnUNet model
        df_lesion_nnunet_3d = pd.read_excel(row['fname_lesion_nnunet_3d'], sheet_name='measures')
        # If the dataframe is empty, insert nan values
        if df_lesion_nnunet_3d.empty:
            df.at[index, 'volume_nnunet_3d'] = np.nan
            df.at[index, 'length_nnunet_3d'] = np.nan
            df.at[index, 'max_axial_damage_ratio_nnunet_3d'] = np.nan
        else:
            # check how many rows are in the dataframe
            if len(df_lesion_nnunet_3d) > 1:
                print(f'WARNING: More than one row in {row["fname_lesion_nnunet_3d"]}')

            # Get volume, length, and max_axial_damage_ratio and save the values in the currently processed row of df
            df.at[index, 'volume_nnunet_3d'] = df_lesion_nnunet_3d['volume [mm3]'].values[0]
            df.at[index, 'length_nnunet_3d'] = df_lesion_nnunet_3d['length [mm]'].values[0]
            df.at[index, 'max_axial_damage_ratio_nnunet_3d'] = df_lesion_nnunet_3d['max_axial_damage_ratio []'].values[0]

        # Read the XLS file with lesion metrics for manual (GT) lesion
        df_lesion_manual = pd.read_excel(row['fname_lesion_manual'], sheet_name='measures')
        # If the dataframe is empty, insert nan values
        if df_lesion_manual.empty:
            df.at[index, 'volume_manual'] = np.nan
            df.at[index, 'length_manual'] = np.nan
            df.at[index, 'max_axial_damage_ratio_manual'] = np.nan
        else:
            # check how many rows are in the dataframe
            if len(df_lesion_manual) > 1:
                print(f'WARNING: More than one row in {row["fname_lesion_manual"]}')

            # Get volume, length, and max_axial_damage_ratio and save the values in the currently processed row of df
            df.at[index, 'volume_manual'] = df_lesion_manual['volume [mm3]'].values[0]
            df.at[index, 'length_manual'] = df_lesion_manual['length [mm]'].values[0]
            df.at[index, 'max_axial_damage_ratio_manual'] = df_lesion_manual['max_axial_damage_ratio []'].values[0]

    print('Here')


if __name__ == '__main__':
    main()
