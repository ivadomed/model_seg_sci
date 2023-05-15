"""
Converts the BIDS-structured sci-zurich AND the augmented Spine-Generic datasets to the nnUNetv2 dataset format. 
Full details about the format can be found here: 
https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md
The (offline) augmentation is done by copying the lesions from SCI-Zurich to the Spine-Generic dataset. Hence, all
the augmented spine-generic subjects are used for training. The validation and test sets strictly consist of SCI-Zurich subjects only.

NOTE: Because this is offline augmentation of an existing dataset, the augmented samples cannot be used for validation. As 
a results, the training/validation splits have to be created manually. This script also creates splits_final.json and writes
in the output directory. 
This file has to be copied to the nnUNet_preprocessed/DatasetXXX_DatasetName after running `nnUNetv2_plan_and_preprocess`
This instructs nnUNet to use the manual splits instead of creating the default 5-fold cross-validation splits.

Note that the conversion from BIDS to nnUNet is done using symbolic links to avoid creating multiple copies of the 
(original) BIDS dataset.

Currently only supports the conversion of a single contrast. In case of multiple contrasts, the script should be 
modified to include those as well. 

Usage example:
    python convert_bids_to_nnUNetv2.py --path-data ~/datasets/sci-zurich --path-out ~/datasets/sci-zurich-nnunet
                    --task-name tSCILesionsZurich --task-number 525 --split 0.8 0.2 --seed 42
"""

import argparse
import pathlib
from pathlib import Path
import json
import os
import glob
from collections import OrderedDict
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

import nibabel as nib
import numpy as np


# parse command line arguments
parser = argparse.ArgumentParser(description='Convert BIDS-structured dataset to nnUNetV2 database format.')
parser.add_argument('--path-data', nargs='+', required=True, type=str,
                    help='Path to BIDS datasets (list).')
parser.add_argument('--path-out', help='Path to output directory.', required=True)
parser.add_argument('--dataset-name', '-dname', default='MSSpineLesion', type=str,
                    help='Specify the task name - usually the anatomy to be segmented, e.g. Hippocampus',)
parser.add_argument('--dataset-number', '-dnum', default=501,type=int, 
                    help='Specify the task number, has to be greater than 500 but less than 999. e.g 502')

parser.add_argument('--seed', default=42, type=int, 
                    help='Seed to be used for the random number generator split into training and test sets.')
# argument that accepts a list of floats as train val test splits
parser.add_argument('--split', nargs='+', required=True, type=float, default=[0.6, 0.2, 0.2],
                    help='Ratios of training, validation and test splits lying between 0-1. Example: --split 0.6 0.2 0.2')

args = parser.parse_args()

# root = Path(args.path_data)
path_out = Path(os.path.join(os.path.abspath(args.path_out), f'Dataset{args.dataset_number}_{args.dataset_name}'))

# create individual directories for train and test images and labels
path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
path_out_imagesTs = Path(os.path.join(path_out, 'imagesTs'))
path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))
path_out_labelsTs = Path(os.path.join(path_out, 'labelsTs'))

train_images, train_labels, test_images, test_labels = [], [], [], []


# creating a manual split for nnunet. Instructions for creating manual splits can be found here:
# https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/manual_data_splits.md
splits_final = [
    {
        'train': [], 
        'val': []
    }
]


def binarize_label(subject_path, label_path):
    label_npy = nib.load(label_path).get_fdata()
    threshold = 0.5
    label_npy = np.where(label_npy > threshold, 1, 0)
    ref = nib.load(subject_path)
    label_bin = nib.Nifti1Image(label_npy, ref.affine, ref.header)
    # overwrite the original label file with the binarized version
    nib.save(label_bin, label_path)


if __name__ == '__main__':

    # make the directories
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTs).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTs).mkdir(parents=True, exist_ok=True)

    # # set the random number generator seed
    # rng = np.random.default_rng(args.seed)

    train_ratio, val_ratio, test_ratio = args.split    
    train_val_subjects = {}
    # loop over the datasets
    for dataset in args.path_data:
        root = Path(dataset)
        
        if "sci-zurich" in dataset:
            logger.info("Found sci-zurich dataset. Converting it into nnUNetv2 format.")
            # Get all subjects from participants.tsv
            subjects_df = pd.read_csv(os.path.join(root, 'participants.tsv'), sep='\t')
            subjects = subjects_df['participant_id'].values.tolist()
            logger.info(f"Total number of subjects in the dataset: {len(subjects)}")

            # Get only the training and test splits initially
            zurich_train_subjects, zurich_test_subjects = train_test_split(subjects, test_size=test_ratio, random_state=args.seed)
            # Use the training split to further split into training and validation splits
            zurich_train_subjects, zurich_val_subjects = train_test_split(zurich_train_subjects, test_size=val_ratio / (train_ratio + val_ratio),
                                                            random_state=args.seed, )
            train_val_subjects['train'] = sorted(zurich_train_subjects)
            train_val_subjects['val'] = sorted(zurich_val_subjects)
            train_val_subjects['zurich_root'] = root
            
            # sort the test subjects
            zurich_test_subjects = sorted(zurich_test_subjects)

        else:
            logger.info("Found augmented spine-generic dataset. Adding to the zurich dataset.")
            # Get the subjects name from the root directory
            subjects_list = [os.path.basename(sub.replace('_0000.nii.gz', '')) for sub in sorted(glob.glob(os.path.join(root, '*.nii.gz')))]
            
            # update the train_val_subjects dictionary with list of subjects 
            train_val_subjects['train'] += subjects_list
            train_val_subjects['sg_root'] = root
            

    train_ctr = 0
    # for data_type, subject in train_val_subjects.items():
    for subject in train_val_subjects['train']:
        
        if subject.startswith('sub-zh'):
            # Another for loop for going through sessions
            temp_subject_path = os.path.join(train_val_subjects['zurich_root'], subject)
            num_sessions_per_subject = sum(os.path.isdir(os.path.join(temp_subject_path, pth)) for pth in os.listdir(temp_subject_path))

            for ses_idx in range(1, num_sessions_per_subject+1):
                train_ctr += 1
                # Get paths with session numbers
                session = 'ses-0' + str(ses_idx)

                subject_images_path = os.path.join(train_val_subjects['zurich_root'], subject, session, 'anat')
                subject_labels_path = os.path.join(train_val_subjects['zurich_root'], 'derivatives', 'labels', subject, session, 'anat')

                subject_image_file = os.path.join(subject_images_path, f"{subject}_{session}_acq-sag_T2w.nii.gz")
                subject_label_file = os.path.join(subject_labels_path, f"{subject}_{session}_acq-sag_T2w_lesion-manual.nii.gz")

                # NOTE: if adding more contrasts, add them here by creating image-label files and the corresponding 
                # nnunet convention names

                # create the new convention names for nnunet
                sub_ses_name = str(Path(subject_image_file).name).split('_')[0] + '_' + str(Path(subject_image_file).name).split('_')[1]
                subject_image_file_nnunet = os.path.join(path_out_imagesTr,f"{args.dataset_name}_{sub_ses_name}_{train_ctr:03d}_0000.nii.gz")
                subject_label_file_nnunet = os.path.join(path_out_labelsTr,f"{args.dataset_name}_{sub_ses_name}_{train_ctr:03d}.nii.gz")

                # add the subject name to the splits_final.json
                splits_final[0]['train'].append(f"{args.dataset_name}_{sub_ses_name}_{train_ctr:03d}")

                # copy the files to new structure using symbolic links (prevents duplication of data and saves space)
                os.symlink(os.path.abspath(subject_image_file), subject_image_file_nnunet)
                os.symlink(os.path.abspath(subject_label_file), subject_label_file_nnunet)

                # binarize the label file
                binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)
        
        else:
            # subject is the augmented version of spine-generic dataset which is already in nnunet format
            train_ctr += 1

            subject_image_file = os.path.join(train_val_subjects['sg_root'], f"{subject}_0000.nii.gz")
            subject_label_file = os.path.join(str(train_val_subjects['sg_root']).replace('imagesTr', 'labelsTr'), f"{subject}.nii.gz")
            
            # print(str(Path(subject_image_file).name).split('_'))

            # create the new convention names for nnunet
            sub_sg = str(Path(subject_image_file).name).split('_')[0]
            sub_ses_zur = str(Path(subject_image_file).name).split('_')[1] + '_' + str(Path(subject_image_file).name).split('_')[2]
            subject_image_file_nnunet = os.path.join(path_out_imagesTr,f"{args.dataset_name}_{sub_sg}_{sub_ses_zur}_{train_ctr:03d}_0000.nii.gz")
            subject_label_file_nnunet = os.path.join(path_out_labelsTr,f"{args.dataset_name}_{sub_sg}_{sub_ses_zur}_{train_ctr:03d}.nii.gz")

            # add the subject name to the splits_final.json
            splits_final[0]['train'].append(f"{args.dataset_name}_{sub_sg}_{sub_ses_zur}_{train_ctr:03d}")

            # copy the files to new structure using symbolic links (prevents duplication of data and saves space)
            os.symlink(os.path.abspath(subject_image_file), subject_image_file_nnunet)
            os.symlink(os.path.abspath(subject_label_file), subject_label_file_nnunet)

            # binarize the label file
            binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)

    # validation subjects 
    val_ctr = 0
    for subject in train_val_subjects['val']:
        if subject.startswith('sub-zh'):
            # Another for loop for going through sessions
            temp_subject_path = os.path.join(train_val_subjects['zurich_root'], subject)
            num_sessions_per_subject = sum(os.path.isdir(os.path.join(temp_subject_path, pth)) for pth in os.listdir(temp_subject_path))

            for ses_idx in range(1, num_sessions_per_subject+1):
                val_ctr += 1
                # Get paths with session numbers
                session = 'ses-0' + str(ses_idx)

                subject_images_path = os.path.join(train_val_subjects['zurich_root'], subject, session, 'anat')
                subject_labels_path = os.path.join(train_val_subjects['zurich_root'], 'derivatives', 'labels', subject, session, 'anat')

                subject_image_file = os.path.join(subject_images_path, f"{subject}_{session}_acq-sag_T2w.nii.gz")
                subject_label_file = os.path.join(subject_labels_path, f"{subject}_{session}_acq-sag_T2w_lesion-manual.nii.gz")

                # NOTE: if adding more contrasts, add them here by creating image-label files and the corresponding 
                # nnunet convention names

                # create the new convention names for nnunet
                sub_ses_name = str(Path(subject_image_file).name).split('_')[0] + '_' + str(Path(subject_image_file).name).split('_')[1]
                subject_image_file_nnunet = os.path.join(path_out_imagesTr,f"{args.dataset_name}_{sub_ses_name}_{train_ctr+val_ctr:03d}_0000.nii.gz")
                subject_label_file_nnunet = os.path.join(path_out_labelsTr,f"{args.dataset_name}_{sub_ses_name}_{train_ctr+val_ctr:03d}.nii.gz")

                # add the subject name to the splits_final.json
                splits_final[0]['val'].append(f"{args.dataset_name}_{sub_ses_name}_{train_ctr+val_ctr:03d}")

                # copy the files to new structure using symbolic links (prevents duplication of data and saves space)
                os.symlink(os.path.abspath(subject_image_file), subject_image_file_nnunet)
                os.symlink(os.path.abspath(subject_label_file), subject_label_file_nnunet)

                # binarize the label file
                binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)
        else:
            print("Subject not in Zurich dataset in the validtation set. Check the splits again.")
            exit()


    test_ctr = 0        
    for subject in zurich_test_subjects:

        # Another for loop for going through sessions
        temp_subject_path = os.path.join(train_val_subjects['zurich_root'], subject)
        num_sessions_per_subject = sum(os.path.isdir(os.path.join(temp_subject_path, pth)) for pth in os.listdir(temp_subject_path))

        for ses_idx in range(1, num_sessions_per_subject+1):
            test_ctr += 1
            # Get paths with session numbers
            session = 'ses-0' + str(ses_idx)

            subject_images_path = os.path.join(train_val_subjects['zurich_root'], subject, session, 'anat')
            subject_labels_path = os.path.join(train_val_subjects['zurich_root'], 'derivatives', 'labels', subject, session, 'anat')                    

            subject_image_file = os.path.join(subject_images_path, f"{subject}_{session}_acq-sag_T2w.nii.gz")
            subject_label_file = os.path.join(subject_labels_path, f"{subject}_{session}_acq-sag_T2w_lesion-manual.nii.gz")

            # NOTE: if adding more contrasts, add them here by creating image-label files and the corresponding 
            # nnunet convention names

            # create the new convention names for nnunet
            sub_ses_name = str(Path(subject_image_file).name).split('_')[0] + '_' + str(Path(subject_image_file).name).split('_')[1]
            subject_image_file_nnunet = os.path.join(path_out_imagesTs,f"{args.dataset_name}_{sub_ses_name}_{test_ctr:03d}_0000.nii.gz")
            subject_label_file_nnunet = os.path.join(path_out_labelsTs,f"{args.dataset_name}_{sub_ses_name}_{test_ctr:03d}.nii.gz")
            
            test_images.append(subject_image_file_nnunet)
            test_labels.append(subject_label_file_nnunet)

            # copy the files to new structure using symbolic links
            os.symlink(os.path.abspath(subject_image_file), subject_image_file_nnunet)
            os.symlink(os.path.abspath(subject_label_file), subject_label_file_nnunet)
            # shutil.copyfile(subject_image_file, subject_image_file_nnunet)
            # shutil.copyfile(subject_label_file, subject_label_file_nnunet)

            # binarize the label file
            binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)
    

    logger.info(f"Number of training and validation subjects (including sessions): {train_ctr}")
    logger.info(f"Number of test subjects (including sessions): {test_ctr}")

    # check whether len of split_json matches train and val ctr
    assert len(splits_final[0]['train']) == train_ctr, 'No. of train images in split_json do not match'
    assert len(splits_final[0]['val']) == val_ctr, 'No. of val images in split_json do not match'
    
    # write the splits_final.json file
    with open(os.path.join(path_out, 'splits_final.json'), 'w') as f:
        json.dump(splits_final, f, indent=4)

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
    json_dict['numTraining'] = (train_ctr + val_ctr)
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


