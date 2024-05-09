"""
Converts the BIDS-structured mslesion_muc datasets to the nnUNetv2 dataset format. 
Full details about the format can be found here: 
https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

The script to be used on a single dataset or multiple datasets.

An option to create region-based labels for segmenting both lesion and the spinal cord is also provided.
Currently only supports the conversion of a single contrast. In case of multiple contrasts, the script should be 
modified to include those as well. 

Usage example single dataset:
    python convert_bids_to_nnUNetv2_all_sci_data.py
        --path-data ~/datasets/mslesion-stitched-muc-rpi
        --path-out ${nnUNet_raw}
        -dname mslesion_stitched_muc
        -dnum 601
        --split 0.8 0.2
        --seed 12
        --region-based

Authors: Julian McGinnis, Naga Karthik, Jan Valosek
"""

import argparse
from pathlib import Path
import json 
import os
import re
import shutil
import yaml
from collections import OrderedDict
from loguru import logger
from sklearn.model_selection import train_test_split
from utils import binarize_label, create_region_based_label, get_git_branch_and_commit, Image
from tqdm import tqdm

import nibabel as nib


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Convert BIDS-structured dataset to nnUNetV2 database format.')
    parser.add_argument('--path-data', nargs='+', required=True, type=str,
                        help='Path to BIDS dataset(s) (list).')
    parser.add_argument('--path-out', help='Path to output directory.', required=True)
    parser.add_argument('--dataset-name', '-dname', default='tSCICombinedRegion', type=str,
                        help='Specify the task name.')
    parser.add_argument('--dataset-number', '-dnum', default=501, type=int,
                        help='Specify the task number, has to be greater than 500 but less than 999. e.g 502')
    parser.add_argument('--seed', default=42, type=int, 
                        help='Seed to be used for the random number generator split into training and test sets.')
    parser.add_argument('--region-based', action='store_true', default=False,
                        help='If set, the script will create labels for region-based nnUNet training. Default: False')
    # argument that accepts a list of floats as train val test splits
    parser.add_argument('--split', nargs='+', required=True, type=float, default=[0.8, 0.2],
                        help='Ratios of training (includes validation) and test splits lying between 0-1. Example: '
                             '--split 0.8 0.2')

    return parser


def get_region_based_label(subject_labels_path, subject_label_file, subject_image_file, sub_ses_name, thr=0.5):
    
    # define path for sc seg file
    subject_seg_file = os.path.join(subject_labels_path, f"{sub_ses_name}_seg-manual.nii.gz")
    
    # check if the seg file exists
    if not os.path.exists(subject_seg_file):
        logger.info(f"Segmentation file for subject {sub_ses_name} does not exist. Skipping.")
        return None

    # create region-based label
    seg_lesion_nii = create_region_based_label(subject_label_file, subject_seg_file, subject_image_file, 
                                               sub_ses_name, thr=thr)
    
    # save the region-based label
    nib.save(seg_lesion_nii, os.path.join(subject_labels_path, f"{sub_ses_name}_seg-lesion-manual.nii.gz"))

    # overwrite the original label file with the region-based label
    subject_label_file = os.path.join(subject_labels_path, f"{sub_ses_name}_seg-lesion-manual.nii.gz")

    return subject_label_file


def main():

    parser = get_parser()
    args = parser.parse_args()

    train_ratio, test_ratio = args.split
    path_out = Path(os.path.join(os.path.abspath(args.path_out), f'Dataset{args.dataset_number}_{args.dataset_name}'))

    # create individual directories for train and test images and labels
    path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
    path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))
    # create the training directories
    Path(path_out).mkdir(parents=True, exist_ok=True)
    Path(path_out_imagesTr).mkdir(parents=True, exist_ok=True)
    Path(path_out_labelsTr).mkdir(parents=True, exist_ok=True)

    # save output to a log file
    logger.add(os.path.join(path_out, "logs.txt"), rotation="10 MB", level="INFO")

    # In case of a single dataset, create test directories only for this dataset
    if len(args.path_data) == 1 and 'zurich' in args.path_data[0]:
        path_out_imagesTsMuc = Path(os.path.join(path_out, 'imagesTsMuc'))
        path_out_labelsTsMuc = Path(os.path.join(path_out, 'labelsTsMuc'))
        Path(path_out_imagesTsMuc).mkdir(parents=True, exist_ok=True)
        Path(path_out_labelsTsMuc).mkdir(parents=True, exist_ok=True)
    # In case of multiple datasets, create test directories for both datasets
    else:
        path_out_imagesTsMuc = Path(os.path.join(path_out, 'imagesTsMuc'))
        path_out_labelsTsMuc = Path(os.path.join(path_out, 'labelsTsMuc'))
        Path(path_out_imagesTsMuc).mkdir(parents=True, exist_ok=True)
        Path(path_out_labelsTsMuc).mkdir(parents=True, exist_ok=True)

    all_subjects, train_subjects, test_subjects = [], {}, {}
    # temp dict for storing dataset commits
    dataset_commits = {}

    # loop over the datasets
    for dataset in args.path_data:
        root = Path(dataset)

        # get the git branch and commit ID of the dataset
        dataset_name = os.path.basename(os.path.normpath(dataset))
        branch, commit = get_git_branch_and_commit(dataset)
        dataset_commits[dataset_name] = f"git-{branch}-{commit}"

        # get the list of subjects in the root directory 
        subjects = [subject for subject in sorted(os.listdir(root)) if subject.startswith('sub-')]

        # add to the list of all subjects
        all_subjects.extend(subjects)
        
        # Get the training and test splits
        tr_subs, te_subs = train_test_split(subjects, test_size=test_ratio, random_state=args.seed)

        # update the train and test subjects dicts with the key as the subject and value as the path to the subject
        train_subjects.update({sub: os.path.join(root, sub) for sub in tr_subs})
        test_subjects.update({sub: os.path.join(root, sub) for sub in te_subs})

    logger.info(f"Total number of subjects combining all datasets: {len(all_subjects)}")
    logger.info(f"Total number of subjects (not images) in the training set (combining all datasets): {len(train_subjects)}")
    logger.info(f"Total number of subjects (not images) in the test set: {len(test_subjects)}")
    print(f"subjects in the training set: {train_subjects.keys()}")

    # print version of each dataset in a separate line
    for dataset_name, dataset_commit in dataset_commits.items():
        logger.info(f"{dataset_name} dataset version: {dataset_commit}")

    train_ctr, test_ctr = 0, 0

    train_niftis, test_nifitis = [], []
    for subject in tqdm(all_subjects, desc="Iterating over all subjects"):

        if subject in train_subjects.keys():

            # check if the subject is from mslesion-tum
            if subject.startswith('sub-m'):
                # Another for loop for going through sessions
                temp_subject_path = train_subjects[subject]
                num_sessions_per_subject = sum(os.path.isdir(os.path.join(temp_subject_path, pth)) for pth in
                                               os.listdir(temp_subject_path))

                for ses_idx in range(1, num_sessions_per_subject+1):    

                    import pdb
                    pdb.set_trace()             
                   
                    train_ctr += 1
                    # Get paths with session numbers
                    session = 'ses-0' + str(ses_idx)

                    subject_images_path = os.path.join(train_subjects[subject], session, 'anat')
                    subject_labels_path = os.path.join(train_subjects[subject].replace(subject, ''), 'derivatives',
                                                    'labels', subject, session, 'anat')

                    # only take the sagittal orientation (as default)
                    subject_image_file = os.path.join(subject_images_path, f"{subject}_{session}_acq-ax_T2w.nii.gz")
                    subject_label_file = os.path.join(subject_labels_path, 
                                                        f"{subject}_{session}_acq-ax_T2w_lesion-manual.nii.gz")

                    # add the subject image file to the list of training niftis
                    train_niftis.append(os.path.basename(subject_image_file))
                    
                    # create the new convention names for nnunet
                    sub_ses_name = f"{str(Path(subject_image_file).name).replace('.nii.gz', '')}"
                    
                    # use region-based labels if required
                    if args.region_based:                        
                        # overwritten the subject_label_file with the region-based label
                        subject_label_file = get_region_based_label(subject_labels_path, subject_label_file, 
                                                                    subject_image_file, sub_ses_name, thr=0.5)
                        if subject_label_file is None:
                            print(f"Skipping since the region-based label could not be generated")
                            continue

                    subject_image_file_nnunet = os.path.join(path_out_imagesTr, 
                                                                f"{args.dataset_name}_{sub_ses_name}_{train_ctr:03d}_0000.nii.gz")
                    subject_label_file_nnunet = os.path.join(path_out_labelsTr,
                                                                f"{args.dataset_name}_{sub_ses_name}_{train_ctr:03d}.nii.gz")
                    
                    # copy the files to new structure using symbolic links (prevents duplication of data and saves space)
                    shutil.copyfile(subject_image_file, subject_image_file_nnunet)
                    shutil.copyfile(subject_label_file, subject_label_file_nnunet)

                    # convert the image and label to RPI using the Image class
                    image = Image(subject_image_file_nnunet)
                    image.change_orientation("RPI")
                    image.save(subject_image_file_nnunet)

                    label = Image(subject_label_file_nnunet)
                    label.change_orientation("RPI")
                    label.save(subject_label_file_nnunet)

                # binarize the label file only if region-based training is not set (since the region-based labels are already binarized)
                if not args.region_based:
                    binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)

            else:
                train_ctr += 1
                subject_images_path = os.path.join(train_subjects[subject], 'anat')
                subject_labels_path = os.path.join(train_subjects[subject].replace(subject, ''), 'derivatives',
                                                   'labels', subject, 'anat')

                subject_image_file = os.path.join(subject_images_path, f"{subject}_T2w.nii.gz")
                subject_label_file = os.path.join(subject_labels_path, f"{subject}_T2w_lesion-manual.nii.gz")

                # add the subject image file to the list of training niftis
                train_niftis.append(os.path.basename(subject_image_file))
                
                # create the new convention names for nnunet
                sub_name = f"{str(Path(subject_image_file).name).replace('.nii.gz', '')}"

                # use region-based labels if required
                if args.region_based:                        
                    # overwritten the subject_label_file with the region-based label
                    subject_label_file = get_region_based_label(subject_labels_path, subject_label_file,
                                                                subject_image_file, sub_name, thr=0.5)
                    if subject_label_file is None:
                        print(f"Skipping since the region-based label could not be generated")
                        continue

                subject_image_file_nnunet = os.path.join(path_out_imagesTr,
                                                         f"{args.dataset_name}_{sub_name}_{train_ctr:03d}_0000.nii.gz")
                subject_label_file_nnunet = os.path.join(path_out_labelsTr,
                                                         f"{args.dataset_name}_{sub_name}_{train_ctr:03d}.nii.gz")
                
                # copy the files to new structure using symbolic links (prevents duplication of data and saves space)
                shutil.copyfile(subject_image_file, subject_image_file_nnunet)
                shutil.copyfile(subject_label_file, subject_label_file_nnunet)

                # convert the image and label to RPI using the Image class
                image = Image(subject_image_file_nnunet)
                image.change_orientation("RPI")
                image.save(subject_image_file_nnunet)

                label = Image(subject_label_file_nnunet)
                label.change_orientation("RPI")
                label.save(subject_label_file_nnunet)

                # binarize the label file only if region-based training is not set (since the region-based labels are already binarized)
                if not args.region_based:
                    binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)

        elif subject in test_subjects:

            if subject.startswith('sub-m'):
                # Another for loop for going through sessions
                temp_subject_path = test_subjects[subject]
                num_sessions_per_subject = sum(os.path.isdir(os.path.join(temp_subject_path, pth)) for pth in os.listdir(temp_subject_path))

                for ses_idx in range(1, num_sessions_per_subject+1):
                    test_ctr_zur += 1
                    # Get paths with session numbers
                    session = 'ses-0' + str(ses_idx)

                    subject_images_path = os.path.join(test_subjects[subject], session, 'anat')
                    subject_labels_path = os.path.join(test_subjects[subject].replace(subject, ''),
                                                       'derivatives', 'labels', subject, session, 'anat')

                    subject_image_file = os.path.join(subject_images_path,
                                                      f"{subject}_{session}_acq-sag_T2w.nii.gz")
                    subject_label_file = os.path.join(subject_labels_path,
                                                      f"{subject}_{session}_acq-sag_T2w_lesion-manual.nii.gz")

                    # add the subject image file to the list of testing niftis
                    test_nifitis.append(os.path.basename(subject_image_file))

                    # create the new convention names for nnunet
                    sub_ses_name = f"{str(Path(subject_image_file).name).replace('.nii.gz', '')}"

                    # use region-based labels if required
                    if args.region_based:                        
                        # overwritten the subject_label_file with the region-based label
                        subject_label_file = get_region_based_label(subject_labels_path, subject_label_file,
                                                                    subject_image_file, sub_ses_name, thr=0.5)
                        if subject_label_file is None:
                            print(f"Skipping since the region-based label could not be generated")
                            continue

                    subject_image_file_nnunet = os.path.join(path_out_imagesTsMuc,
                                                             f"{args.dataset_name}_{sub_ses_name}_{test_ctr_zur:03d}_0000.nii.gz")
                    subject_label_file_nnunet = os.path.join(path_out_labelsTsMuc,
                                                             f"{args.dataset_name}_{sub_ses_name}_{test_ctr_zur:03d}.nii.gz")
                    
                    # copy the files to new structure using symbolic links
                    shutil.copyfile(subject_image_file, subject_image_file_nnunet)
                    shutil.copyfile(subject_label_file, subject_label_file_nnunet)

                    # convert the image and label to RPI using the Image class
                    image = Image(subject_image_file_nnunet)
                    image.change_orientation("RPI")
                    image.save(subject_image_file_nnunet)

                    label = Image(subject_label_file_nnunet)
                    label.change_orientation("RPI")
                    label.save(subject_label_file_nnunet)

                    # binarize the label file only if region-based training is not set (since the region-based labels are already binarized)
                    if not args.region_based:
                        binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)
            
            else:
                print(f"Skipping Subject {subject} as it is from sci-paris")
                continue

        else:
            print("Skipping file, could not be located in the Train or Test splits split.", subject)

    logger.info(f"----- Dataset conversion finished! -----")
    logger.info(f"Number of training and validation images (not subjects) including sessions: {train_ctr}")
    logger.info(f"Number of test images (not subjects) including sessions in SCI-Munich: {test_ctr}")

    # assert train_ctr == len(train_subjects), 'No. of train/val images do not match'
    # assert test_ctr == len(test_subjects), 'No. of test images do not match'

    # create a yaml file containing the list of training and test niftis
    niftis_dict = {
        f"train": sorted(train_niftis),
        f"test": sorted(test_nifitis)
    }

    # write the train and test niftis to a yaml file
    with open(f"dataset_split_seed{args.seed}.yaml", "w") as outfile:
        yaml.dump(niftis_dict, outfile, default_flow_style=False)

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
    json_dict['seed_used'] = args.seed
    json_dict['dataset_versions'] = dataset_commits
    json_dict['image_orientation'] = "RPI"
    
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
        0: "acq-ax_T2w",
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
