"""
Converts the BIDS-structured sci-zurich dataset to the nnUNetv2 dataset format. Full details about 
the format can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

Note that the conversion from BIDS to nnUNet is done using symbolic links to avoid creating multiple copies of the 
(original) BIDS dataset.

Currently only supports the conversion of a single contrast. In case of multiple contrasts, the script should be 
modified to include those as well. 

Usage example:
    python convert_bids_to_nnUNetv2.py --path-data ~/datasets/sci-zurich --path-out ~/datasets/sci-zurich-nnunet
                    --dataset-name tSCILesionsZurich --dataset-number 501 --split 0.6 0.2 0.2 --seed 99

    python convert_bids_to_nnUNetv2.py --path-data ~/datasets/sci-zurich --path-out ~/datasets/sci-zurich-nnunet
                --dataset-name tSCILesionsZurich --dataset-number 501 --split 0.8 0.2 --seed 99
"""

import argparse
import pathlib
from pathlib import Path
import json
import os
from collections import OrderedDict
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

import nibabel as nib
import numpy as np


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Convert BIDS-structured dataset to nnUNetV2 database format.')
    parser.add_argument('--path-data', help='Path to BIDS dataset.', required=True)
    parser.add_argument('--path-out', help='Path to output directory.', required=True)
    parser.add_argument('--dataset-name', '-dname', default='tSCILesionsZurich', type=str,
                        help='Specify the task name.')
    parser.add_argument('--dataset-number', '-dnum', default=501, type=int,
                        help='Specify the task number, has to be greater than 500 but less than 999. e.g 502')
    parser.add_argument('--seed', default=99, type=int,
                        help='Seed to be used for the random number generator split into training and test sets.')
    # argument that accepts a list of floats as train val test splits
    parser.add_argument('--split', nargs='+', required=True, type=float, default=[0.6, 0.2, 0.2],
                        help='Ratios of training/validation/test splits lying between 0-1. Use 3 values if creating'
                             'a dummy dataset (Example: --split 0.6 0.2 0.2). Else use only train/test split '
                             '(Example: --split 0.8 0.2)')
    parser.add_argument('--include-masks_folders', action='store_true', default=False,
                        help='Include masks folders (with SC segmentations) in the output dataset. Default: False')
    parser.add_argument('--create-dummy-dataset', '-dummy', action='store_true', default=False,
                        help='Create a dummy dataset with training subjects only. Default: False')

    return parser


def binarize_label(subject_path, label_path):
    label_npy = nib.load(label_path).get_fdata()
    # NOTE: using a very small threshold (<<< 0) to binarize the label leads to more 
    # more volume of the label being retained. For e.g. due to PVE, the voxels which have 
    # value of 0.0001 in the label file will still be retained in the binarized label as 1.
    # Since this is not a correct representation of the label, we use a threshold of 0.5.
    threshold = 0.5
    label_npy = np.where(label_npy > threshold, 1, 0)
    ref = nib.load(subject_path)
    label_bin = nib.Nifti1Image(label_npy, ref.affine, ref.header)
    # overwrite the original label file with the binarized version
    nib.save(label_bin, label_path)


def main():

    parser = get_parser()
    args = parser.parse_args()

    root = Path(os.path.abspath(os.path.expanduser(args.path_data)))
    path_out = Path(os.path.join(os.path.abspath(os.path.expanduser(args.path_out)),
                                 f'Dataset{args.dataset_number}_{args.dataset_name}'))

    # create individual directories for train and test images and labels
    path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
    path_out_imagesTs = Path(os.path.join(path_out, 'imagesTs'))
    path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))
    path_out_labelsTs = Path(os.path.join(path_out, 'labelsTs'))

    train_images, train_labels, train_masks, test_images, test_labels, test_masks = [], [], [], [], [], []

    # make the directories
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTs).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTs).mkdir(parents=True, exist_ok=True)

    if args.include_masks_folders:
        # create masks directories with SC masks
        path_out_masksTr = Path(os.path.join(path_out, 'masksTr'))
        path_out_masksTs = Path(os.path.join(path_out, 'masksTs'))
        pathlib.Path(path_out_masksTr).mkdir(parents=True, exist_ok=True)
        pathlib.Path(path_out_masksTs).mkdir(parents=True, exist_ok=True)

    # set the random number generator seed
    rng = np.random.default_rng(args.seed)

    # Get all subjects from participants.tsv
    subjects_df = pd.read_csv(os.path.join(root, 'participants.tsv'), sep='\t')
    subjects = subjects_df['participant_id'].values.tolist()
    logger.info(f"Total number of subjects in the dataset: {len(subjects)}")

    if args.create_dummy_dataset:
        assert len(args.split) == 3, 'The split argument must have 3 values for train, val and test splits. E.g. ' \
                                     '[0.6 0.2 0.2]'
        train_ratio, val_ratio, test_ratio = args.split
        # Get the training and test splits
        train_subjects, test_subjects = train_test_split(subjects, test_size=test_ratio, random_state=args.seed)
        # Use the training split to further split into training and validation splits
        train_subjects, val_subjects = train_test_split(train_subjects, test_size=val_ratio / (train_ratio + val_ratio),
                                                        random_state=args.seed)
        logger.info(f"Creating a dummy dataset with {len(train_subjects)} training subjects only.")
    else:
        assert len(args.split) == 2, 'The split argument must have 2 values for train and test splits. E.g. ' \
                                     '[0.8 0.2]. If you indeed want to use 3 values, use also the ' \
                                     '--create-dummy-dataset flag.'

        train_ratio, test_ratio = args.split
        # Get the training and test splits
        train_subjects, test_subjects = train_test_split(subjects, test_size=test_ratio, random_state=args.seed)
        logger.info(f"Creating a dataset with {len(train_subjects)} training subjects and {len(test_subjects)} test subjects.")


    # print(train_subjects[:4])

    train_ctr, test_ctr = 0, 0
    for subject in subjects:

        if subject in train_subjects:

            # Another for loop for going through sessions
            temp_subject_path = os.path.join(root, subject)
            num_sessions_per_subject = sum(os.path.isdir(os.path.join(temp_subject_path, pth)) for pth in os.listdir(temp_subject_path))

            for ses_idx in range(1, num_sessions_per_subject+1):
                train_ctr += 1
                # Get paths with session numbers
                session = 'ses-0' + str(ses_idx)

                subject_images_path = os.path.join(root, subject, session, 'anat')
                subject_labels_path = os.path.join(root, 'derivatives', 'labels', subject, session, 'anat')                    

                subject_image_file = os.path.join(subject_images_path, f"{subject}_{session}_acq-sag_T2w.nii.gz")
                subject_label_file = os.path.join(subject_labels_path, f"{subject}_{session}_acq-sag_T2w_lesion-manual.nii.gz")

                # NOTE: if adding more contrasts, add them here by creating image-label files and the corresponding 
                # nnunet convention names

                # create the new convention names for nnunet
                sub_ses_name = str(Path(subject_image_file).name).split('_')[0] + '_' + str(Path(subject_image_file).name).split('_')[1]
                subject_image_file_nnunet = os.path.join(path_out_imagesTr,
                                                         f"{sub_ses_name}_{train_ctr:03d}_0000.nii.gz")
                subject_label_file_nnunet = os.path.join(path_out_labelsTr,
                                                         f"{sub_ses_name}_{train_ctr:03d}.nii.gz")
                
                train_images.append(subject_image_file_nnunet)
                train_labels.append(subject_label_file_nnunet)

                # copy the files to new structure using symbolic links (prevents duplication of data and saves space)
                os.symlink(os.path.abspath(subject_image_file), subject_image_file_nnunet)
                os.symlink(os.path.abspath(subject_label_file), subject_label_file_nnunet)

                # binarize the label file
                binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)

                if args.include_masks_folders:
                    subject_mask_file = os.path.join(subject_labels_path,
                                                     f"{subject}_{session}_acq-sag_T2w_seg-manual.nii.gz")
                    subject_mask_file_nnunet = os.path.join(path_out_masksTr, f"{sub_ses_name}_{train_ctr:03d}.nii.gz")
                    train_masks.append(subject_mask_file_nnunet)
                    if os.path.isfile(subject_mask_file):
                        os.symlink(os.path.abspath(subject_mask_file), subject_mask_file_nnunet)
                    else:
                        print(f"Mask file {subject_mask_file} not found. Skipping this subject.")
        elif subject in test_subjects:

            # Another for loop for going through sessions
            temp_subject_path = os.path.join(root, subject)
            num_sessions_per_subject = sum(os.path.isdir(os.path.join(temp_subject_path, pth)) for pth in os.listdir(temp_subject_path))

            for ses_idx in range(1, num_sessions_per_subject+1):
                test_ctr += 1
                # Get paths with session numbers
                session = 'ses-0' + str(ses_idx)

                subject_images_path = os.path.join(root, subject, session, 'anat')
                subject_labels_path = os.path.join(root, 'derivatives', 'labels', subject, session, 'anat')

                subject_image_file = os.path.join(subject_images_path, f"{subject}_{session}_acq-sag_T2w.nii.gz")
                subject_label_file = os.path.join(subject_labels_path, f"{subject}_{session}_acq-sag_T2w_lesion-manual.nii.gz")

                # NOTE: if adding more contrasts, add them here by creating image-label files and the corresponding
                # nnunet convention names

                # create the new convention names for nnunet
                sub_ses_name = str(Path(subject_image_file).name).split('_')[0] + '_' + str(Path(subject_image_file).name).split('_')[1]
                subject_image_file_nnunet = os.path.join(path_out_imagesTs,
                                                         f"{sub_ses_name}_{test_ctr:03d}_0000.nii.gz")
                subject_label_file_nnunet = os.path.join(path_out_labelsTs,
                                                         f"{sub_ses_name}_{test_ctr:03d}.nii.gz")

                test_images.append(subject_image_file_nnunet)
                test_labels.append(subject_label_file_nnunet)

                # copy the files to new structure using symbolic links
                os.symlink(os.path.abspath(subject_image_file), subject_image_file_nnunet)
                os.symlink(os.path.abspath(subject_label_file), subject_label_file_nnunet)
                # shutil.copyfile(subject_image_file, subject_image_file_nnunet)
                # shutil.copyfile(subject_label_file, subject_label_file_nnunet)

                # binarize the label file
                binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)

                if args.include_masks_folders:
                    subject_mask_file = os.path.join(subject_labels_path,
                                                     f"{subject}_{session}_acq-sag_T2w_seg-manual.nii.gz")
                    subject_mask_file_nnunet = os.path.join(path_out_masksTs, f"{sub_ses_name}_{test_ctr:03d}.nii.gz")
                    test_masks.append(subject_mask_file_nnunet)
                    if os.path.isfile(subject_mask_file):
                        os.symlink(os.path.abspath(subject_mask_file), subject_mask_file_nnunet)
                    else:
                        print(f"Mask file {subject_mask_file} not found. Skipping this subject.")
        else:
            if args.create_dummy_dataset:
                print("Skipping file, as it is in the validation split.", subject)
            else:
                print("Skipping file, could not be located in the Train or Test splits split.", subject)

    logger.info(f"Number of training and validation subjects (including sessions): {train_ctr}")
    logger.info(f"Number of test subjects (including sessions): {test_ctr}")
    # assert train_ctr == len(train_subjects), 'No. of train/val images do not match'
    # assert test_ctr == len(test_subjects), 'No. of test images do not match'

    # c.f. dataset json generation
    # In nnUNet V2, dataset.json file has become much shorter. The description of the fields and changes
    # can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md#datasetjson
    # this file can be automatically generated using the following code here:
    # https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/dataset_conversion/generate_dataset_json.py
    # example: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task055_SegTHOR.py

    json_dict = OrderedDict()
    json_dict['name'] = args.dataset_name
    json_dict['description'] = args.dataset_name
    json_dict['reference'] = "TBD"
    json_dict['licence'] = "TBD"
    json_dict['release'] = "0.0"
    json_dict['numTraining'] = train_ctr
    json_dict['numTest'] = test_ctr

    # The following keys are the most important ones. 
    """
    channel_names:
        Channel names must map the index to the name of the channel. For BIDS, this refers to the contrast suffix.
        {
            0: 'T1',
            1: 'CT'
        }
    Note that the channel names may influence the normalization scheme!! Learn more in the documentation.

    labels:
        This will tell nnU-Net what labels to expect. Important: This will also determine whether you use region-based training or not.
        Example regular labels:
        {
            'background': 0,
            'left atrium': 1,
            'some other label': 2
        }
        Example region-based training: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/region_based_training.md
        {
            'background': 0,
            'whole tumor': (1, 2, 3),
            'tumor core': (2, 3),
            'enhancing tumor': 3
        }
        Remember that nnU-Net expects consecutive values for labels! nnU-Net also expects 0 to be background!
    """

    json_dict['channel_names'] = {
        0: "acq-sag_T2w",
    }

    json_dict['labels'] = {
        "background": 0,
        "lesion": 1,
    }
    
    # Needed for finding the files correctly. IMPORTANT! File endings must match between images and segmentations!
    json_dict['file_ending'] = ".nii.gz"

    # create dataset_description.json
    json_object = json.dumps(json_dict, indent=4)
    # write to dataset description
    # nn-unet requires it to be "dataset.json"
    dataset_dict_name = f"dataset.json"
    with open(os.path.join(path_out, dataset_dict_name), "w") as outfile:
        outfile.write(json_object)


if __name__ == '__main__':
    main()
