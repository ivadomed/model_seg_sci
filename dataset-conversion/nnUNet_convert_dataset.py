"""
Converts the BIDS-structured sci-zurich dataset to the nnUNet dataset format. Full details about 
the format can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/documentation/dataset_conversion.md

Note that the conversion from BIDS to nnUNet is done using symbolic links to avoid creating multiple copies of the 
(original) BIDS dataset.

Currently only supports the conversion of a single contrast. In case of multiple contrasts, the script should be 
modified to include those as well. 

Usage example:
    python nnUNet_convert_dataset.py --path-data ~/datasets/sci-zurich --path-out ~/datasets/sci-zurich-nnuent
                    --task-name tSCILesionsZurich --task-number 525 --split 0.8 0.2 --seed 42
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
parser = argparse.ArgumentParser(description='Convert BIDS-structured dataset to nnUNet database format.')
parser.add_argument('--path-data', help='Path to BIDS dataset.', required=True)
parser.add_argument('--path-out', help='Path to output directory.', required=True)
parser.add_argument('--task-name', default='MSSpineLesion', type=str,
                    help='Specify the task name - usually the anatomy to be segmented, e.g. Hippocampus',)
parser.add_argument('--task-number', default=501,type=int, 
                    help='Specify the task number, has to be greater than 500 but less than 999. e.g 502')

parser.add_argument('--seed', default=42, type=int, 
                    help='Seed to be used for the random number generator split into training and test sets.')
# argument that accepts a list of floats as train val test splits
parser.add_argument('--split', nargs='+', required=True, type=float, default=[0.8, 0.2],
                    help='Ratios of training (includes validation) and test splits lying between 0-1. Example: --split 0.8 0.2')

args = parser.parse_args()

root = Path(args.path_data)
train_ratio, test_ratio = args.split
path_out = Path(os.path.join(os.path.abspath(args.path_out), f'Task{args.task_number}_{args.task_name}'))

# create individual directories for train and test images and labels
path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
path_out_imagesTs = Path(os.path.join(path_out, 'imagesTs'))
path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))
path_out_labelsTs = Path(os.path.join(path_out, 'labelsTs'))

train_images, train_labels, test_images, test_labels = [], [], [], []


def binarize_label(subject_path, label_path):
    label_npy = nib.load(label_path).get_fdata()
    threshold = 1e-12
    label_npy = np.where(label_npy > threshold, 1, 0)
    ref = nib.load(subject_path)
    label_bin = nib.Nifti1Image(label_npy, ref.affine, ref.header)
    # overwrite the original label file with the binarized version
    nib.save(label_bin, label_path)

# multi_session = not args.single_session


if __name__ == '__main__':

    # make the directories
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTs).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTs).mkdir(parents=True, exist_ok=True)

    # set the random number generator seed
    rng = np.random.default_rng(args.seed)

    # Get all subjects from participants.tsv
    subjects_df = pd.read_csv(os.path.join(root, 'participants.tsv'), sep='\t')
    subjects = subjects_df['participant_id'].values.tolist()
    logger.info(f"Total number of subjects in the dataset: {len(subjects)}")

    # Get the training and test splits
    train_subjects, test_subjects = train_test_split(subjects, test_size=test_ratio, random_state=args.seed)
    rng.shuffle(train_subjects)

    train_ctr, test_ctr = 0, 0
    for subject in subjects:

        if subject in train_subjects:
            # check if a "ses" directory exists
            if os.path.isdir(os.path.join(root, subject, 'ses-01')):
                train_ctr += 1
                subject_images_path = os.path.join(root, subject, 'ses-01', 'anat')
                subject_labels_path = os.path.join(root, 'derivatives', 'labels', subject, 'ses-01', 'anat')

                subject_image_file = os.path.join(subject_images_path, f"{subject}_ses-01_acq-sag_T2w.nii.gz")
                subject_label_file = os.path.join(subject_labels_path, f"{subject}_ses-01_acq-sag_T2w_lesion-manual.nii.gz")

                # NOTE: if adding more contrasts, add them here by creating image-label files and the corresponding 
                # nnunet convention names

                # create the new convention names for nnunet
                sub_ses_name = str(Path(subject_image_file).name).split('_')[0] + '_' + str(Path(subject_image_file).name).split('_')[1]
                subject_image_file_nnunet = os.path.join(path_out_imagesTr,f"{args.task_name}_{sub_ses_name}_{train_ctr:03d}_0000.nii.gz")
                subject_label_file_nnunet = os.path.join(path_out_labelsTr,f"{args.task_name}_{sub_ses_name}_{train_ctr:03d}_0000.nii.gz")
                
                train_images.append(subject_image_file_nnunet)
                train_labels.append(subject_label_file_nnunet)

                # copy the files to new structure using symbolic links (prevents duplication of data and saves space)
                os.symlink(os.path.abspath(subject_image_file), subject_image_file_nnunet)
                os.symlink(os.path.abspath(subject_label_file), subject_label_file_nnunet)

                # binarize the label file
                binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)

                # conversion_dict[str(os.path.abspath(ax_file))] = ax_file_nnunet
                # conversion_dict[str(os.path.abspath(seg_file))] = seg_file_nnunet
        
        elif subject in test_subjects:
            if os.path.isdir(os.path.join(root, subject, 'ses-01')):
                test_ctr += 1
                subject_images_path = os.path.join(root, subject, 'ses-01', 'anat')
                subject_labels_path = os.path.join(root, 'derivatives', 'labels', subject, 'ses-01', 'anat')

                subject_image_file = os.path.join(subject_images_path, f"{subject}_ses-01_acq-sag_T2w.nii.gz")
                subject_label_file = os.path.join(subject_labels_path, f"{subject}_ses-01_acq-sag_T2w_lesion-manual.nii.gz")

                # NOTE: if adding more contrasts, add them here by creating image-label files and the corresponding 
                # nnunet convention names

                # create the new convention names for nnunet
                sub_ses_name = str(Path(subject_image_file).name).split('_')[0] + '_' + str(Path(subject_image_file).name).split('_')[1]
                subject_image_file_nnunet = os.path.join(path_out_imagesTs,f"{args.task_name}_{sub_ses_name}_{test_ctr:03d}_0000.nii.gz")
                subject_label_file_nnunet = os.path.join(path_out_labelsTs,f"{args.task_name}_{sub_ses_name}_{test_ctr:03d}_0000.nii.gz")
                
                test_images.append(subject_image_file_nnunet)
                test_labels.append(subject_label_file_nnunet)

                # copy the files to new structure using symbolic links
                os.symlink(os.path.abspath(subject_image_file), subject_image_file_nnunet)
                os.symlink(os.path.abspath(subject_label_file), subject_label_file_nnunet)
                # shutil.copyfile(subject_image_file, subject_image_file_nnunet)
                # shutil.copyfile(subject_label_file, subject_label_file_nnunet)

                # binarize the label file
                binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)
        
        else:
            print("Skipping file, could not be located in the Train or Test splits split.", subject)


    assert train_ctr == len(train_subjects), 'No. of train/val images do not match'
    assert test_ctr == len(test_subjects), 'No. of test images do not match'

    # # create dataset_description.json
    # json_object = json.dumps(conversion_dict, indent=4)
    # # write to dataset description
    # conversion_dict_name = f"conversion_dict_sagittal_channel_{args.use_sag_channel}.json"
    # with open(os.path.join(path_out, conversion_dict_name), "w") as outfile:
    #     outfile.write(json_object)


    # c.f. dataset json generation
    # general info : https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/utils.py
    # example: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task055_SegTHOR.py

    json_dict = OrderedDict()
    json_dict['name'] = args.task_name
    json_dict['description'] = args.task_name
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "TBD"
    json_dict['licence'] = "TBD"
    json_dict['release'] = "0.0"

    json_dict['modality'] = {
        "0": "acq-sag_T2w",
        }
    
    json_dict['labels'] = {
        "0": "background",
        "1": "lesion",
        }
    json_dict['numTraining'] = train_ctr
    json_dict['numTest'] = test_ctr

    json_dict['training'] = [{
        "image": str(train_labels[i]).replace("labelsTr", "imagesTr") , 
        "label": train_labels[i] 
        } for i in range(len(train_images))]
    # Note: See https://github.com/MIC-DKFZ/nnUNet/issues/407 for how this should be described
    json_dict['test'] = [str(test_labels[i]).replace("labelsTs", "imagesTs") for i in range(len(test_images))]

    # create dataset_description.json
    json_object = json.dumps(json_dict, indent=4)
    # write to dataset description
    # nn-unet requires it to be "dataset.json"
    dataset_dict_name = f"dataset.json"
    with open(os.path.join(path_out, dataset_dict_name), "w") as outfile:
        outfile.write(json_object)


