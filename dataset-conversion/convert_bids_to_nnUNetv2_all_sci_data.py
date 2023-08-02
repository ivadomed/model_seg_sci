"""
Converts the BIDS-structured sci-zurich, sci-colorado, and sci-paris datasets to the nnUNetv2 dataset format. 
Full details about the format can be found here: 
https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

An option to create region-based labels for segmentating both lesion and the spinal cord is also provided. 
Currently only supports the conversion of a single contrast. In case of multiple contrasts, the script should be 
modified to include those as well. 

Usage example:
    python convert_bids_to_nnUNetv2_all_sci_data.py 
        --path-data ~/datasets/sci-zurich-rpi ~/datasets/sci-colorado-rpi ~/datasets/sci-paris-rpi 
        --path-out ${nnUNet_raw}
        -dname tSCICombinedRegion
        -dnum 275
        --split 0.8 0.2
        --seed 50
        --region-based

Authors: Naga Karthik
"""

import argparse
from pathlib import Path
import json 
import os
import re
import shutil
from collections import OrderedDict
from loguru import logger
from sklearn.model_selection import train_test_split
from utils import binarize_label, create_region_based_label

import nibabel as nib
import numpy as np


def get_parser():
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
    parser.add_argument('--region-based', action='store_true', default=False,
                        help='If set, the script will create labels for region-based nnUNet training. Default: False')
    # argument that accepts a list of floats as train val test splits
    parser.add_argument('--split', nargs='+', required=True, type=float, default=[0.8, 0.2],
                        help='Ratios of training (includes validation) and test splits lying between 0-1. Example: --split 0.8 0.2')

    return parser



def get_region_based_label(subject_labels_path, subject_label_file, subject_image_file, sub_ses_name, thr=0.5):
    
    if sub_ses_name.startswith('sub-zh'):
        # define path for sc seg file
        subject_seg_file = os.path.join(subject_labels_path, f"{sub_ses_name}_acq-sag_T2w_seg-manual.nii.gz")
    else:
        subject_seg_file = os.path.join(subject_labels_path, f"{sub_ses_name}_T2w_seg-manual.nii.gz")
    
    # check if the seg file exists
    if not os.path.exists(subject_seg_file):
        logger.info(f"Segmentation file for subject {sub_ses_name} does not exist. Skipping.")
        return None

    # create region-based label
    seg_lesion_nii = create_region_based_label(subject_label_file, subject_seg_file, subject_image_file, sub_ses_name, thr=thr)
    
    if sub_ses_name.startswith('sub-zh'):
        # save the region-based label
        nib.save(seg_lesion_nii, os.path.join(subject_labels_path, f"{sub_ses_name}_acq-sag_T2w_seg-lesion-manual.nii.gz"))

        # overwrite the original label file with the region-based label
        subject_label_file = os.path.join(subject_labels_path, f"{sub_ses_name}_acq-sag_T2w_seg-lesion-manual.nii.gz")
    else:
        # save the region-based label
        nib.save(seg_lesion_nii, os.path.join(subject_labels_path, f"{sub_ses_name}_T2w_seg-lesion-manual.nii.gz"))

        # overwrite the original label file with the region-based label
        subject_label_file = os.path.join(subject_labels_path, f"{sub_ses_name}_T2w_seg-lesion-manual.nii.gz")

    return subject_label_file


def main():

    parser = get_parser()
    args = parser.parse_args()

    train_ratio, test_ratio = args.split
    path_out = Path(os.path.join(os.path.abspath(args.path_out), f'Dataset{args.dataset_number}_{args.dataset_name}'))

    # create individual directories for train and test images and labels
    path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
    path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))

    path_out_imagesTsZur = Path(os.path.join(path_out, 'imagesTsZur'))
    path_out_labelsTsZur = Path(os.path.join(path_out, 'labelsTsZur'))

    path_out_imagesTsCol = Path(os.path.join(path_out, 'imagesTsCol'))
    path_out_labelsTsCol = Path(os.path.join(path_out, 'labelsTsCol'))

    # make the directories
    Path(path_out).mkdir(parents=True, exist_ok=True)
    Path(path_out_imagesTr).mkdir(parents=True, exist_ok=True)
    Path(path_out_labelsTr).mkdir(parents=True, exist_ok=True)
    
    Path(path_out_imagesTsZur).mkdir(parents=True, exist_ok=True)
    Path(path_out_labelsTsZur).mkdir(parents=True, exist_ok=True)

    Path(path_out_imagesTsCol).mkdir(parents=True, exist_ok=True)
    Path(path_out_labelsTsCol).mkdir(parents=True, exist_ok=True)

    all_subjects, train_subjects, test_subjects = [], {}, {}
    # loop over the datasets
    for dataset in args.path_data:
        root = Path(dataset)
        
        if "sci-paris" not in dataset:
            # get the list of subjects in the root directory 
            subjects = [subject for subject in sorted(os.listdir(root)) if subject.startswith('sub-')]

            # add to the list of all subjects
            all_subjects.extend(subjects)
            
            # Get the training and test splits
            tr_subs, te_subs = train_test_split(subjects, test_size=test_ratio, random_state=args.seed)
            # update the train and test subjects dicts with the key as the subject and value as the path to the subject
            train_subjects.update({sub: os.path.join(root, sub) for sub in tr_subs})
            test_subjects.update({sub: os.path.join(root, sub) for sub in te_subs})
        else:
            # get the list of subjects in the root directory 
            subjects = [subject for subject in sorted(os.listdir(root)) if subject.startswith('sub-')]
            # logger.info(f"Total number of subjects in the Colorado dataset: {len(subjects)}")

            # add to the list of all subjects
            all_subjects.extend(subjects)

            # add all the subjects to train_subjects dict
            train_subjects.update({sub: os.path.join(root, sub) for sub in subjects})
        
    # print(f"Total number of subjects in the dataset: {len(all_subjects)}")
    # print(f"Total number of subjects in the training set: {len(train_subjects)}")
    # print(f"Total number of subjects in the test set: {len(test_subjects)}")
    # print(f"subjects in the training set: {train_subjects.keys()}")

    train_ctr, test_ctr_zur, test_ctr_col = 0, 0, 0
    for subject in all_subjects:

        if subject in train_subjects.keys():

            # check if the subject is from sci-zurich
            if subject.startswith('sub-zh'):
                # Another for loop for going through sessions
                temp_subject_path = train_subjects[subject]
                num_sessions_per_subject = sum(os.path.isdir(os.path.join(temp_subject_path, pth)) for pth in os.listdir(temp_subject_path))

                for ses_idx in range(1, num_sessions_per_subject+1):
                    train_ctr += 1
                    # Get paths with session numbers
                    session = 'ses-0' + str(ses_idx)

                    subject_images_path = os.path.join(train_subjects[subject], session, 'anat')
                    subject_labels_path = os.path.join(train_subjects[subject].replace(subject, ''), 'derivatives', 'labels', subject, session, 'anat')                    

                    subject_image_file = os.path.join(subject_images_path, f"{subject}_{session}_acq-sag_T2w.nii.gz")
                    subject_label_file = os.path.join(subject_labels_path, f"{subject}_{session}_acq-sag_T2w_lesion-manual.nii.gz")

                    # NOTE: if adding more contrasts, add them here by creating image-label files and the corresponding 
                    # nnunet convention names

                    # create the new convention names for nnunet
                    sub_ses_name = str(Path(subject_image_file).name).split('_')[0] + '_' + str(Path(subject_image_file).name).split('_')[1]
                    
                    # use region-based labels if required
                    if args.region_based:                        
                        # overwritten the subject_label_file with the region-based label
                        subject_label_file = get_region_based_label(subject_labels_path, subject_label_file, subject_image_file, sub_ses_name, thr=0.5)
                        if subject_label_file is None:
                            print(f"Skipping since the region-based label could not be generated")
                            continue

                    subject_image_file_nnunet = os.path.join(path_out_imagesTr,f"{args.dataset_name}_{sub_ses_name}_{train_ctr:03d}_0000.nii.gz")
                    subject_label_file_nnunet = os.path.join(path_out_labelsTr,f"{args.dataset_name}_{sub_ses_name}_{train_ctr:03d}.nii.gz")
                    
                    # copy the files to new structure using symbolic links (prevents duplication of data and saves space)
                    shutil.copyfile(subject_image_file, subject_image_file_nnunet)
                    shutil.copyfile(subject_label_file, subject_label_file_nnunet)

                    # binarize the label file only if region-based training is not set (since the region-based labels are already binarized)
                    if not args.region_based:
                        binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)
            
            # subject is from sci-colorado or sci-paris
            else:
                train_ctr += 1
                subject_images_path = os.path.join(train_subjects[subject], 'anat')
                subject_labels_path = os.path.join(train_subjects[subject].replace(subject, ''), 'derivatives', 'labels', subject, 'anat')                    

                subject_image_file = os.path.join(subject_images_path, f"{subject}_T2w.nii.gz")
                subject_label_file = os.path.join(subject_labels_path, f"{subject}_T2w_lesion-manual.nii.gz")
                # NOTE: if adding more contrasts, add them here by creating image-label files and the corresponding 
                # nnunet convention names

                # create the new convention names for nnunet
                sub_name = str(Path(subject_image_file).name).split('_')[0]

                # use region-based labels if required
                if args.region_based:                        
                    # overwritten the subject_label_file with the region-based label
                    subject_label_file = get_region_based_label(subject_labels_path, subject_label_file, subject_image_file, sub_name, thr=0.5)
                    if subject_label_file is None:
                        print(f"Skipping since the region-based label could not be generated")
                        continue

                subject_image_file_nnunet = os.path.join(path_out_imagesTr,f"{args.dataset_name}_{sub_name}_{train_ctr:03d}_0000.nii.gz")
                subject_label_file_nnunet = os.path.join(path_out_labelsTr,f"{args.dataset_name}_{sub_name}_{train_ctr:03d}.nii.gz")
                
                # copy the files to new structure using symbolic links (prevents duplication of data and saves space)
                shutil.copyfile(subject_image_file, subject_image_file_nnunet)
                shutil.copyfile(subject_label_file, subject_label_file_nnunet)

                # binarize the label file only if region-based training is not set (since the region-based labels are already binarized)
                if not args.region_based:
                    binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)
            

        elif subject in test_subjects:

            if subject.startswith('sub-zh'):
                # Another for loop for going through sessions
                temp_subject_path = test_subjects[subject]
                num_sessions_per_subject = sum(os.path.isdir(os.path.join(temp_subject_path, pth)) for pth in os.listdir(temp_subject_path))

                for ses_idx in range(1, num_sessions_per_subject+1):
                    test_ctr_zur += 1
                    # Get paths with session numbers
                    session = 'ses-0' + str(ses_idx)

                    subject_images_path = os.path.join(test_subjects[subject], session, 'anat')
                    subject_labels_path = os.path.join(test_subjects[subject].replace(subject, ''), 'derivatives', 'labels', subject, session, 'anat')                    

                    subject_image_file = os.path.join(subject_images_path, f"{subject}_{session}_acq-sag_T2w.nii.gz")
                    subject_label_file = os.path.join(subject_labels_path, f"{subject}_{session}_acq-sag_T2w_lesion-manual.nii.gz")

                    # create the new convention names for nnunet
                    sub_ses_name = str(Path(subject_image_file).name).split('_')[0] + '_' + str(Path(subject_image_file).name).split('_')[1]

                    # use region-based labels if required
                    if args.region_based:                        
                        # overwritten the subject_label_file with the region-based label
                        subject_label_file = get_region_based_label(subject_labels_path, subject_label_file, subject_image_file, sub_ses_name, thr=0.5)
                        if subject_label_file is None:
                            print(f"Skipping since the region-based label could not be generated")
                            continue

                    subject_image_file_nnunet = os.path.join(path_out_imagesTsZur,f"{args.dataset_name}_{sub_ses_name}_{test_ctr_zur:03d}_0000.nii.gz")
                    subject_label_file_nnunet = os.path.join(path_out_labelsTsZur,f"{args.dataset_name}_{sub_ses_name}_{test_ctr_zur:03d}.nii.gz")
                    
                    # copy the files to new structure using symbolic links
                    shutil.copyfile(subject_image_file, subject_image_file_nnunet)
                    shutil.copyfile(subject_label_file, subject_label_file_nnunet)

                    # binarize the label file only if region-based training is not set (since the region-based labels are already binarized)
                    if not args.region_based:
                        binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)

            elif re.match(r'sub-\d{4}', subject):
                test_ctr_col += 1
                subject_images_path = os.path.join(test_subjects[subject], 'anat')
                subject_labels_path = os.path.join(test_subjects[subject].replace(subject, ''), 'derivatives', 'labels', subject, 'anat')                    

                subject_image_file = os.path.join(subject_images_path, f"{subject}_T2w.nii.gz")
                subject_label_file = os.path.join(subject_labels_path, f"{subject}_T2w_lesion-manual.nii.gz")

                # create the new convention names for nnunet
                sub_name = str(Path(subject_image_file).name).split('_')[0] # + '_' + str(Path(subject_image_file).name).split('_')[1]

                # use region-based labels if required
                if args.region_based:                        
                    # overwritten the subject_label_file with the region-based label
                    subject_label_file = get_region_based_label(subject_labels_path, subject_label_file, subject_image_file, sub_name, thr=0.5)
                    if subject_label_file is None:
                        continue

                subject_image_file_nnunet = os.path.join(path_out_imagesTsCol,f"{args.dataset_name}_{sub_name}_{test_ctr_col:03d}_0000.nii.gz")
                subject_label_file_nnunet = os.path.join(path_out_labelsTsCol,f"{args.dataset_name}_{sub_name}_{test_ctr_col:03d}.nii.gz")
                
                # copy the files to new structure using symbolic links
                shutil.copyfile(subject_image_file, subject_image_file_nnunet)
                shutil.copyfile(subject_label_file, subject_label_file_nnunet)

                # binarize the label file only if region-based training is not set (since the region-based labels are already binarized)
                if not args.region_based:
                    binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)
            
            else:
                print(f"Skipping Subject {subject} as it is from sci-paris")
                continue

        else:
            print("Skipping file, could not be located in the Train or Test splits split.", subject)

    logger.info(f"----- Dataset conversion finished! -----")
    logger.info(f"Number of training and validation images (including sessions): {train_ctr}")
    logger.info(f"Number of test subjects (including sessions) in SCI-Zurich: {test_ctr_zur}")
    logger.info(f"Number of test subjects (including sessions) in SCI-Colorado: {test_ctr_col}")
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
    json_dict['numTest'] = test_ctr_zur + test_ctr_col
    json_dict['seed_used'] = args.seed
    
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

    if not args.region_based:
        json_dict['labels'] = {
            "background": 0,
            "lesion": 1,
        }
    else:
        json_dict['labels'] = {
            "background": 0,
            "sc": [1, 2],
            # "sc": 1,
            "lesion": 2,
        }
        json_dict['regions_class_order'] = [1, 2]
    
    # Needed for finding the files correctly. IMPORTANT! File endings must match between images and segmentations!
    json_dict['file_ending'] = ".nii.gz"

    # create dataset_description.json
    json_object = json.dumps(json_dict, indent=4)
    # write to dataset description
    # nn-unet requires it to be "dataset.json"
    dataset_dict_name = f"dataset.json"
    with open(os.path.join(path_out, dataset_dict_name), "w") as outfile:
        outfile.write(json_object)

if __name__ == "__main__":
    main()