#!/usr/bin/env python
#
# Script to perform manual correction of segmentations and vertebral labeling.
#
# First, make sure that FSLeyes is installed and can be called without having 
# to activate the FSLeyes conda environment:
#   sudo ln -s $(command -v fsleyes) /usr/local/bin/fsleyes
# Then, activate SCT's Python environment:
#   source ${SCT_DIR}/python/etc/profile.d/conda.sh
#   conda activate venv_sct
# 
# For usage, type: python manual_correction.py -h
#
# Authors: Jan Valosek, Julien Cohen-Adad, Sandrine Bédard
#
# Notes:
#     - This script is a modified version of the script used in the ukbiobank-spinalcord-csa project.
#       (https://github.com/sct-pipeline/ukbiobank-spinalcord-csa/blob/master/pipeline_ukbiobank/cli/manual_correction.py)
#     - There is also currently an open issue in SCT to incorporate this functionality into an 
#       SCT script. (https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3353)

import argparse
import glob
import json
import logging
import os
import sys
import shutil
import subprocess
import time
import yaml

# Folder where to output manual labels, at the root of a BIDS dataset.
# TODO: make it an input argument (with default value)
FOLDER_DERIVATIVES = os.path.join('derivatives', 'labels')

logging.basicConfig(stream=sys.stdout, level='INFO', format="%(levelname)s %(message)s")


def get_parser():
    """
    parser function
    """
    parser = argparse.ArgumentParser(
        description='Manual correction of spinal cord segmentation, vertebral and pontomedullary junction labeling. '
                    'Manually corrected files are saved under derivatives/ folder (BIDS standard).',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-config',
        metavar="<file>",
        required=True,
        help=
        "Config yaml file listing images that require manual corrections for segmentation and vertebral "
        "labeling. 'FILES_SEG' lists images associated with spinal cord segmentation "
        ",'FILES_LABEL' lists images associated with vertebral labeling "
        "and 'FILES_PMJ' lists images associated with pontomedullary junction labeling"
        "You can validate your .yml file at this website: http://www.yamllint.com/."
        " If you want to correct segmentation only, ommit 'FILES_LABEL' in the list. Below is an example .yml file:"
        + 
            """
            FILES_SEG: \n
            - sub-1000032_T1w.nii.gz
            - sub-1000083_T2w.nii.gz
            FILES_LABEL:
            - sub-1000032_T1w.nii.gz
            - sub-1000710_T1w.nii.gz
            FILES_PMJ:
            - sub-1000032_T1w.nii.gz
            - sub-1000710_T1w.nii.gz\n
            """
    )
    parser.add_argument(
        '-path-in',
        metavar="<folder>",
        help='Path to the processed data. Example: output/data_processed',
        default='./'
    )
    parser.add_argument(
        '-path-out',
        metavar="<folder>",
        help="Path to the BIDS dataset where the corrected labels will be generated. Note: if the derivatives/ folder "
             "does not already exist, it will be created."
             "Example: ~/julien/mydataset",
        default='./'
    )
    parser.add_argument(
        '-qc-only',
        help="Only output QC report based on the manually-corrected files already present in the derivatives folder. "
             "Skip the copy of the source files, and the opening of the manual correction pop-up windows.",
        action='store_true'
    )
    parser.add_argument(
        '-add-seg-only',
        help="Only copy the source files (segmentation) that aren't in -config list to the derivatives/ folder. "
             "Use this flag to add manually QC-ed automatic segmentations to the derivatives folder.",
        action='store_true'
    )
    parser.add_argument(
        '-v', '--verbose',
        help="Full verbose (for debugging)",
        action='store_true'
    )

    return parser

# BIDS utility tool
def get_subject(file):
    """
    Get subject from BIDS file name
    :param file:
    :return: subject
    """
    return file.split('_')[0]


def get_contrast(file):
    """
    Get contrast from BIDS file name
    :param file:
    :return:
    """
    return 'dwi' if (file.split('_')[-1]).split('.')[0] == 'dwi' else 'anat'


def splitext(fname):
        """
        Split a fname (folder/file + ext) into a folder/file and extension.

        Note: for .nii.gz the extension is understandably .nii.gz, not .gz
        (``os.path.splitext()`` would want to do the latter, hence the special case).
        """
        dir, filename = os.path.split(fname)
        for special_ext in ['.nii.gz', '.tar.gz']:
            if filename.endswith(special_ext):
                stem, ext = filename[:-len(special_ext)], special_ext
                return os.path.join(dir, stem), ext
        # If no special case, behaves like the regular splitext
        stem, ext = os.path.splitext(filename)
        return os.path.join(dir, stem), ext


def add_suffix(fname, suffix):
    """
    Add suffix between end of file name and extension.

    :param fname: absolute or relative file name. Example: t2.nii
    :param suffix: suffix. Example: _mean
    :return: file name with suffix. Example: t2_mean.nii

    Examples:

    - add_suffix(t2.nii, _mean) -> t2_mean.nii
    - add_suffix(t2.nii.gz, a) -> t2a.nii.gz
    """
    stem, ext = splitext(fname)
    return os.path.join(stem + suffix + ext)


def remove_suffix(fname, suffix):
    """
    Remove suffix between end of file name and extension.

    :param fname: absolute or relative file name with suffix. Example: t2_mean.nii
    :param suffix: suffix. Example: _mean
    :return: file name without suffix. Example: t2.nii

    Examples:

    - remove_suffix(t2_mean.nii, _mean) -> t2.nii
    - remove_suffix(t2a.nii.gz, a) -> t2.nii.gz
    """
    stem, ext = splitext(fname)
    return os.path.join(stem.replace(suffix, '') + ext)


def curate_dict_yml(dict_yml):
    """
    Curate dict_yml to only have filenames instead of absolute path
    :param dict_yml: dict: input yml file as dict
    :return: dict_yml_curate
    """
    dict_yml_curate = {}
    for task, files in dict_yml.items():
        dict_yml_curate[task] = [os.path.basename(file) for file in files]
    return dict_yml_curate


def check_files_exist(dict_files, path_data):
    """
    Check if all files listed in the input dictionary exist
    :param dict_files:
    :param path_data: folder where BIDS dataset is located
    :return:
    """
    missing_files = []
    for task, files in dict_files.items():
        if files is not None:
            for file in files:
                fname = os.path.join(path_data, get_subject(file), get_contrast(file), file)
                if not os.path.exists(fname):
                    missing_files.append(fname)
    if missing_files:
        logging.error("The following files are missing: \n{}. \nPlease check that the files listed "
                        "in the yaml file and the input path are correct.".format(missing_files))


def check_output_folder(path_bids, folder_derivatives):
    """
    Make sure path exists, has writing permissions, and create derivatives folder if it does not exist.
    :param path_bids:
    :return: path_bids_derivatives
    """
    if path_bids is None:
        logging.error("-path-out should be provided.")
    if not os.path.exists(path_bids):
        logging.error("Output path does not exist: {}".format(path_bids))
    path_bids_derivatives = os.path.join(path_bids, folder_derivatives)
    os.makedirs(path_bids_derivatives, exist_ok=True)
    return path_bids_derivatives


def check_software_installed(list_software=['sct', 'fsleyes']):
    """
    Make sure software are installed
    :param list_software: {'sct'}
    :return:
    """
    install_ok = True
    software_cmd = {
        'sct': 'sct_version',
        'fsleyes': 'fsleyes --version'
        }
    logging.info("Checking if required software are installed...")
    for software in list_software:
        try:
            output = subprocess.check_output(software_cmd[software], shell=True)
            logging.info("'{}' (version: {}) is installed.".format(software, output.decode('utf-8').strip('\n')))
        except:
            logging.error("'{}' is not installed. Please install it before using this program.".format(software))
            install_ok = False
    return install_ok


def get_function(task):
    if task == 'FILES_SEG':
        return 'sct_deepseg_sc'
    elif task == 'FILES_LABEL':
        return 'sct_label_utils'
    elif task == 'FILES_PMJ':
        return 'sct_detect_pmj'
    else:
        raise ValueError("This task is not recognized: {}".format(task))


def get_suffix(task, suffix=''):
    if task == 'FILES_SEG':
        return '_seg'+suffix
    elif task == 'FILES_LABEL':
        return '_labels'+suffix
    elif task == 'FILES_PMJ':
        return '_pmj'+suffix

    else:
        raise ValueError("This task is not recognized: {}".format(task))


def correct_segmentation(fname, fname_seg_out):
    """
    Copy fname_seg in fname_seg_out, then open FSLeyes with fname and fname_seg_out.
    :param fname:
    :param fname_seg:
    :param fname_seg_out:
    :param name_rater:
    :return:
    """
    # launch FSLeyes
    print("In FSLeyes, click on 'Edit mode', correct the segmentation, then save it with the same name (overwrite).")
    os.system('fsleyes -yh ' + fname + ' ' + fname_seg_out + ' -cm red')


def correct_vertebral_labeling(fname, fname_label):
    """
    Open sct_label_utils to manually label vertebral levels.
    :param fname:
    :param fname_label:
    :param name_rater:
    :return:
    """
    message = "Click at the posterior tip of the disc between C1-C2, C2-C3 and C3-C4 vertebral levels, then click 'Save and Quit'."
    os.system('sct_label_utils -i {} -create-viewer 2,3,4 -o {} -msg "{}"'.format(fname, fname_label, message))


def correct_pmj_label(fname, fname_label):
    """
    Open sct_label_utils to manually label PMJ.
    :param fname:
    :param fname_label:
    :param name_rater:
    :return:
    """
    message = "Click at the posterior tip of the pontomedullary junction (PMJ) then click 'Save and Quit'."
    os.system('sct_label_utils -i {} -create-viewer 50 -o {} -msg "{}"'.format(fname, fname_label, message))


def create_json(fname_nifti, name_rater):
    """
    Create json sidecar with meta information
    :param fname_nifti: str: File name of the nifti image to associate with the json sidecar
    :param name_rater: str: Name of the expert rater
    :return:
    """
    metadata = {'Author': name_rater, 'Date': time.strftime('%Y-%m-%d %H:%M:%S')}
    fname_json = fname_nifti.rstrip('.nii').rstrip('.nii.gz') + '.json'
    with open(fname_json, 'w') as outfile:
        json.dump(metadata, outfile, indent=4)


def main():

    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Check if required software is installed (skip that if -qc-only is true)
    if not args.qc_only:
        if not check_software_installed():
            sys.exit("Some required software are not installed. Exit program.")

    # check if input yml file exists
    if os.path.isfile(args.config):
        fname_yml = args.config
    else:
        sys.exit("ERROR: Input yml file {} does not exist or path is wrong.".format(args.config))

    # fetch input yml file as dict
    with open(fname_yml, 'r') as stream:
        try:
            dict_yml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Curate dict_yml to only have filenames instead of absolute path
    dict_yml = curate_dict_yml(dict_yml)

    # check for missing files before starting the whole process
    check_files_exist(dict_yml, args.path_in)

    # check that output folder exists and has write permission
    path_out_deriv = check_output_folder(args.path_out, FOLDER_DERIVATIVES)

    # Get name of expert rater (skip if -qc-only is true)
    if not args.qc_only:
        name_rater = input("Enter your name (Firstname Lastname). It will be used to generate a json sidecar with each "
                           "corrected file: ")

    # Build QC report folder name
    fname_qc = 'qc_corr_' + time.strftime('%Y%m%d%H%M%S')

    # Get list of segmentations files for all subjects in -path-in (if -add-seg-only)
    if args.add_seg_only:
        path_list = glob.glob(args.path_in + "/**/*_seg.nii.gz", recursive=True)  # TODO: add other extension
        # Get only filenames without suffix _seg  to match files in -config .yml list
        file_list = [remove_suffix(os.path.split(path)[-1], '_seg') for path in path_list]

    # TODO: address "none" issue if no file present under a key
    # Perform manual corrections
    for task, files in dict_yml.items():
        # Get the list of segmentation files to add to derivatives, excluding the manually corrrected files in -config.
        if args.add_seg_only and task == 'FILES_SEG':
            # Remove the files in the -config list
            for file in files:
                if file in file_list:
                    file_list.remove(file)
            files = file_list  # Rename to use those files instead of the ones to exclude
        if files is not None:
            for file in files:
                # build file names
                subject = file.split('_')[0]
                contrast = get_contrast(file)
                fname = os.path.join(args.path_in, subject, contrast, file)
                fname_label = os.path.join(
                    path_out_deriv, subject, contrast, add_suffix(file, get_suffix(task, '-manual')))
                os.makedirs(os.path.join(path_out_deriv, subject, contrast), exist_ok=True)
                if not args.qc_only:
                    if os.path.isfile(fname_label):
                        # if corrected file already exists, asks user if they want to overwrite it
                        answer = None
                        while answer not in ("y", "n"):
                            answer = input("WARNING! The file {} already exists. "
                                           "Would you like to modify it? [y/n] ".format(fname_label))
                            if answer == "y":
                                do_labeling = True
                                overwrite = False
                            elif answer == "n":
                                do_labeling = False
                            else:
                                print("Please answer with 'y' or 'n'")
                    else:
                        do_labeling = True
                        overwrite = True
                    # Perform labeling for the specific task
                    if do_labeling:
                        if task in ['FILES_SEG']:
                            fname_seg = add_suffix(fname, get_suffix(task))
                            if overwrite:
                                shutil.copyfile(fname_seg, fname_label)
                            if not args.add_seg_only:
                                correct_segmentation(fname, fname_label)
                        elif task == 'FILES_LABEL':
                            correct_vertebral_labeling(fname, fname_label)
                        elif task == 'FILES_PMJ':
                            correct_pmj_label(fname, fname_label)
                        else:
                            sys.exit('Task not recognized from yml file: {}'.format(task))
                        # create json sidecar with the name of the expert rater
                        create_json(fname_label, name_rater)

                # generate QC report (only for vertebral labeling or for qc only)
                if args.qc_only or task != 'FILES_SEG':
                    os.system('sct_qc -i {} -s {} -p {} -qc {} -qc-subject {}'.format(
                        fname, fname_label, get_function(task), fname_qc, subject))
                    # Archive QC folder
                    shutil.copy(fname_yml, fname_qc)
                    shutil.make_archive(fname_qc, 'zip', fname_qc)
                    print("Archive created:\n--> {}".format(fname_qc+'.zip'))


if __name__ == '__main__':
    main()
