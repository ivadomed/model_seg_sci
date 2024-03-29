"""
Converts the BIDS-structured spine-generic data-multi-subject dataset to the nnUNetv2 dataset format. Full details about
the format can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

Note that the conversion from BIDS to nnUNet is done using symbolic links to avoid creating multiple copies of the
(original) BIDS dataset.

NOTE: There is no split. We are using this script just to convert the dataset to the nnUNet format to do the lesion
augmentation. The split is then done by another script.

Usage example:
    python convert_bids_to_nnUNetv2_spine-generic.py --path-data ~/datasets/data-multi-subject --path-out ~/datasets/data-multi-subject-nnunet
                    --dataset-name SpineGenericMutliSubject --dataset-number 526 --seed 99
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

# parse command line arguments
parser = argparse.ArgumentParser(description='Convert BIDS-structured dataset to nnUNetV2 database format. No split is '
                                             'done -- the whole dataset is used.')
parser.add_argument('--path-data', help='Path to BIDS dataset.', required=True)
parser.add_argument('--path-out', help='Path to output directory.', required=True)
parser.add_argument('--dataset-name', '-dname', default='SpineGenericMutliSubject', type=str,
                    help='Specify the task name.')
parser.add_argument('--dataset-number', '-dnum', default=501, type=int,
                    help='Specify the task number, has to be greater than 500 but less than 999. e.g 502')
parser.add_argument('--seed', default=99, type=int,
                    help='Seed to be used for the random number generator split into training and test sets.')

args = parser.parse_args()

root = Path(os.path.abspath(os.path.expanduser(args.path_data)))
path_out = Path(os.path.join(os.path.abspath(os.path.expanduser(args.path_out)), f'Dataset{args.dataset_number}_{args.dataset_name}'))

# create individual directories for train and test images and labels
path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))        # labels will be create by the augmentation
# create masks directories with SC masks
path_out_masksTr = Path(os.path.join(path_out, 'masksTr'))

train_images, train_labels, train_masks, test_images, test_labels, test_masks = [], [], [], [], [], []


if __name__ == '__main__':

    # make the directories
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_masksTr).mkdir(parents=True, exist_ok=True)

    # set the random number generator seed
    rng = np.random.default_rng(args.seed)

    # Get all subjects from participants.tsv
    subjects_df = pd.read_csv(os.path.join(root, 'participants.tsv'), sep='\t')
    subjects = subjects_df['participant_id'].values.tolist()
    logger.info(f"Total number of subjects in the dataset: {len(subjects)}")

    # Using all spine-generic subjects for the augmentation during training (i.e., no split)
    train_subjects = subjects

    train_ctr, test_ctr = 0, 0
    for subject in subjects:

        if subject in train_subjects:

            train_ctr += 1

            subject_images_path = os.path.join(root, subject, 'anat')
            subject_labels_path = os.path.join(root, 'derivatives', 'labels', subject, 'anat')

            subject_image_file = os.path.join(subject_images_path, f"{subject}_T2w.nii.gz")
            subject_mask_file = os.path.join(subject_labels_path, f"{subject}_T2w_seg-manual.nii.gz")

            # NOTE: if adding more contrasts, add them here by creating image-label files and the corresponding
            # nnunet convention names

            # create the new convention names for nnunet
            sub_name = str(Path(subject_image_file).name).split('_')[0]
            subject_image_file_nnunet = os.path.join(path_out_imagesTr,
                                                     f"{args.dataset_name}_{sub_name}_{train_ctr:03d}_0000.nii.gz")
            subject_mask_file_nnunet = os.path.join(path_out_masksTr,
                                                    f"{args.dataset_name}_{sub_name}_{train_ctr:03d}.nii.gz")

            train_images.append(subject_image_file_nnunet)
            train_masks.append(subject_mask_file_nnunet)

            # copy the files to new structure using symbolic links (prevents duplication of data and saves space)
            os.symlink(os.path.abspath(subject_image_file), subject_image_file_nnunet)
            os.symlink(os.path.abspath(subject_mask_file), subject_mask_file_nnunet)


    logger.info(f"Number of training and validation subjects (including sessions): {train_ctr}")
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
