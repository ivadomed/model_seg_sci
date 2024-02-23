"""
Convert BIDS-structured PRAXIS datasets (e.g., site-003, site-012, ...) to the nnUNetv2 MULTI-CHANNEL format.
Note: site-013 and site-014 have only 4 and 5 subjects, respectively, with usable T2w sag images. Thus, we use them
ONLY for testing, NOT for training.

dataset.json:

```json
    "channel_names": {
        "0": "acq-sag_T2w"
    },
    "labels": {
        "background": 0,
        "sc": [
            1,
            2
        ],
        "lesion": 2
    },
    "regions_class_order": [
        1,
        2
    ],
```

Full details about the format can be found here:
https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

The script to be used on a single dataset or multiple datasets.

An option to create region-based labels for segmenting both lesion and the spinal cord is also provided.
Currently only supports the conversion of a single contrast. In case of multiple contrasts, the script should be
modified to include those as well.

# Note: the script performs RPI reorientation of the images and labels

Usage example multiple datasets:
    python convert_bids_to_nnUNetv2_praxis_region-based.py
        --path-data ~/data/site-003 ~/data/site-012 ~/data/site-013 ~/data/site-014
        --path-out ${nnUNet_raw}
        -dname tSCIpraxis
        -dnum 275
        --split 0.8 0.2
        --seed 50
        --region-based

Usage example single dataset:
    python convert_bids_to_nnUNetv2_praxis_region-based.py
        --path-data ~/data/site-003
        --path-out ${nnUNet_raw}
        -dname tSCIpraxis
        -dnum 275
        --split 0.8 0.2
        --seed 50
        --region-based

Authors: Naga Karthik, Jan Valosek
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

# site-013 and site-014 have only 4 and 5 subjects, respectively, with usable T2w sag images.
# Thus, we use them ONLY for testing, NOT for training.
LIST_OF_TESTING_SUBJECTS = ['sub-ham027_acq-sag_run-01_T2w_lesion.nii.gz',
                            'sub-ham089_acq-sag_run-01_T2w_lesion.nii.gz',
                            'sub-ham196_acq-sag_run-01_T2w_lesion.nii.gz',
                            'sub-ham349_acq-sag_run-01_T2w_lesion.nii.gz',
                            'sub-que002_acq-sagittal_run-01_T2w_lesion.nii.gz',
                            'sub-que004_acq-sagittal_run-04_T2w_lesion.nii.gz',
                            'sub-que008_acq-sagittal_run-01_T2w_lesion.nii.gz',
                            'sub-que011_acq-sagittal_T2w_lesion.nii.gz',
                            'sub-que012_acq-sagittal_run-02_T2w_lesion.nii.gz']


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Convert BIDS-structured dataset to nnUNetV2 REGION-BASED format.')
    parser.add_argument('--path-data', nargs='+', required=True, type=str,
                        help='Path to BIDS dataset(s) (list).')
    parser.add_argument('--path-out', help='Path to output directory.', required=True)
    parser.add_argument('--dataset-name', '-dname', default='tSCIpraxis', type=str,
                        help='Specify the task name.')
    parser.add_argument('--dataset-number', '-dnum', default=501, type=int,
                        help='Specify the task number, has to be greater than 500 but less than 999. e.g 502')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to be used for the random number generator split into training and test sets.')
    parser.add_argument('--region-based', action='store_true', default=False,
                        help='If set, the script will create labels for region-based nnUNet training. Default: False')
    # argument that accepts a list of floats as train val test splits
    parser.add_argument('--split', nargs='+', type=float, default=[0.8, 0.2],
                        help='Ratios of training (includes validation) and test splits lying between 0-1. Example: '
                             '--split 0.8 0.2')
    return parser


def get_region_based_label(subject_label_file, subject_image_file, sub_ses_name, thr=0.5):
    # define path for sc seg file
    subject_seg_file = subject_label_file.replace('_lesion', '_seg')

    # check if the seg file exists
    if not os.path.exists(subject_seg_file):
        logger.info(f"Spinal cord segmentation file for subject {sub_ses_name} does not exist. Skipping.")
        return None

    # create region-based label
    seg_lesion_nii = create_region_based_label(subject_label_file, subject_seg_file, subject_image_file,
                                               sub_ses_name, thr=thr)

    # save the region-based label
    combined_seg_file = subject_label_file.replace('_lesion', '_seg-lesion')
    nib.save(seg_lesion_nii, combined_seg_file)

    return combined_seg_file


def create_directories(path_out, site):
    """Create test directories for a specified site.

    Args:
    path_out (str): Base output directory.
    site (str): Site identifier, such as 'site-001', 'site-002', etc.
    """
    paths = [Path(path_out, f'imagesTs_{site}'),
             Path(path_out, f'labelsTs_{site}')]

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def find_site_in_path(path):
    """Extracts site identifier from the given path.

    Args:
    path (str): Input path containing a site identifier.

    Returns:
    str: Extracted site identifier or None if not found.
    """
    match = re.search(r'site_\d{3}', path)
    return match.group(0) if match else None


def create_yaml(train_niftis, test_nifitis, path_out, args, train_ctr, test_ctr, dataset_commits):
    # create a yaml file containing the list of training and test niftis
    niftis_dict = {
        f"train": sorted(train_niftis),
        f"test": sorted(test_nifitis)
    }

    # write the train and test niftis to a yaml file
    with open(os.path.join(path_out, f"train_test_split_seed{args.seed}.yaml"), "w") as outfile:
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


def main():
    parser = get_parser()
    args = parser.parse_args()

    train_ratio, test_ratio = args.split
    if args.region_based:
        args.dataset_name = f"{args.dataset_name}RegionBased"

    path_out = Path(os.path.join(os.path.abspath(args.path_out), f'Dataset{args.dataset_number}_{args.dataset_name}'
                                                                 f'Seed{args.seed}'))

    # create individual directories for train and test images and labels
    path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
    path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))
    # create the training directories
    Path(path_out).mkdir(parents=True, exist_ok=True)
    Path(path_out_imagesTr).mkdir(parents=True, exist_ok=True)
    Path(path_out_labelsTr).mkdir(parents=True, exist_ok=True)

    # save output to a log file
    logger.add(os.path.join(path_out, "logs.txt"), rotation="10 MB", level="INFO")

    # Check if dataset paths exist
    for path in args.path_data:
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist.")

    # Get sites from the input paths
    sites = set(find_site_in_path(path) for path in args.path_data if find_site_in_path(path))
    # Single site
    if len(sites) == 1:
        create_directories(path_out, sites.pop())
    # Multiple sites
    else:
        for site in sites:
            create_directories(path_out, site)

    all_lesion_files, train_subjects, test_subjects = [], {}, {}
    # temp dict for storing dataset commits
    dataset_commits = {}

    # loop over the datasets
    for dataset in args.path_data:
        root = Path(dataset)

        # get the git branch and commit ID of the dataset
        dataset_name = os.path.basename(os.path.normpath(dataset))
        branch, commit = get_git_branch_and_commit(dataset)
        dataset_commits[dataset_name] = f"git-{branch}-{commit}"

        if dataset_name not in ['site_013', 'site_014']:
            # get recursively all GT '_lesion' files
            lesion_files = [str(path) for path in root.rglob('*_lesion.nii.gz')]

            # add to the list of all subjects
            all_lesion_files.extend(lesion_files)

            # Get the training and test splits
            tr_subs, te_subs = train_test_split(lesion_files, test_size=test_ratio, random_state=args.seed)

        # Add images from site_013 and site_014 to the test set -- these sites have only 4 and 5 subjects, respectively,
        # with usable T2w sag images. Thus, we use them ONLY for testing, NOT for training.
        elif dataset_name in ['site_013', 'site_014']:
            te_subs = []
            for subject in LIST_OF_TESTING_SUBJECTS:
                te_subs.extend([str(path) for path in root.rglob(subject)])

            # add to the list of all subjects
            all_lesion_files.extend(te_subs)

        # update the train and test subjects dicts with the key as the subject and value as the path to the subject
        train_subjects.update({sub: os.path.join(root, sub) for sub in tr_subs})
        test_subjects.update({sub: os.path.join(root, sub) for sub in te_subs})

    logger.info(f"Found subjects in the training set (combining all datasets): {len(train_subjects)}")
    logger.info(f"Found subjects in the test set (combining all datasets): {len(test_subjects)}")
    # Print test subjects for each site
    for site in sites:
        logger.info(f"Test subjects in {site}: {len([sub for sub in test_subjects if site in sub])}")

    # print version of each dataset in a separate line
    for dataset_name, dataset_commit in dataset_commits.items():
        logger.info(f"{dataset_name} dataset version: {dataset_commit}")

    # Counters for train and test sets
    train_ctr, test_ctr = 0, 0
    train_niftis, test_nifitis = [], []
    # Loop over all subjects
    for subject_label_file in tqdm(all_lesion_files, desc="Iterating over all subjects"):

        # Construct path to the background image
        subject_image_file = subject_label_file.replace('/derivatives/labels', '').replace('_lesion', '')

        # Train subjects
        if subject_label_file in train_subjects.keys():

            train_ctr += 1
            # add the subject image file to the list of training niftis
            train_niftis.append(os.path.basename(subject_image_file))

            # create the new convention names for nnunet
            sub_name = f"{str(Path(subject_image_file).name).replace('.nii.gz', '')}"

            subject_image_file_nnunet = os.path.join(path_out_imagesTr,
                                                     f"{args.dataset_name}_{sub_name}_{train_ctr:03d}_0000.nii.gz")
            subject_label_file_nnunet = os.path.join(path_out_labelsTr,
                                                     f"{args.dataset_name}_{sub_name}_{train_ctr:03d}.nii.gz")

            # use region-based labels if required
            if args.region_based:
                # overwritten the subject_label_file with the region-based label
                subject_label_file = get_region_based_label(subject_label_file,
                                                            subject_image_file, sub_name, thr=0.5)
                if subject_label_file is None:
                    print(f"Skipping since the region-based label could not be generated")
                    continue

            # copy the files to new structure
            shutil.copyfile(subject_image_file, subject_image_file_nnunet)
            shutil.copyfile(subject_label_file, subject_label_file_nnunet)

            # convert the image and label to RPI using the Image class
            image = Image(subject_image_file_nnunet)
            image.change_orientation("RPI")
            image.save(subject_image_file_nnunet)

            label = Image(subject_label_file_nnunet)
            label.change_orientation("RPI")
            label.save(subject_label_file_nnunet)

            # binarize the label file only if region-based training is not set (since the region-based labels are
            # already binarized)
            if not args.region_based:
                binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)

        # Test subjects
        elif subject_label_file in test_subjects:

            test_ctr += 1
            # add the subject image file to the list of testing niftis
            test_nifitis.append(os.path.basename(subject_image_file))

            # create the new convention names for nnunet
            sub_name = f"{str(Path(subject_image_file).name).replace('.nii.gz', '')}"

            subject_image_file_nnunet = os.path.join(Path(path_out,
                                                          f'imagesTs_{find_site_in_path(test_subjects[subject_label_file])}'),
                                                     f'{args.dataset_name}_{sub_name}_{test_ctr:03d}_0000.nii.gz')
            subject_label_file_nnunet = os.path.join(Path(path_out,
                                                          f'labelsTs_{find_site_in_path(test_subjects[subject_label_file])}'),
                                                     f'{args.dataset_name}_{sub_name}_{test_ctr:03d}.nii.gz')

            # use region-based labels if required
            if args.region_based and find_site_in_path(test_subjects[subject_label_file]) != 'site_014':
                # overwritten the subject_label_file with the region-based label
                subject_label_file = get_region_based_label(subject_label_file,
                                                            subject_image_file, sub_name, thr=0.5)
                if subject_label_file is None:
                    continue

                shutil.copyfile(subject_label_file, subject_label_file_nnunet)
                label = Image(subject_label_file_nnunet)
                label.change_orientation("RPI")
                label.save(subject_label_file_nnunet)

            # copy the files to new structure
            shutil.copyfile(subject_image_file, subject_image_file_nnunet)
            print(f"\nCopying {subject_image_file} to {subject_image_file_nnunet}")

            # convert the image and label to RPI using the Image class
            image = Image(subject_image_file_nnunet)
            image.change_orientation("RPI")
            image.save(subject_image_file_nnunet)

            # binarize the label file only if region-based training is not set (since the region-based labels are
            # already binarized)
            if not args.region_based:
                binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)

        else:
            print("Skipping file, could not be located in the Train or Test splits split.", subject_label_file)

    logger.info(f"----- Dataset conversion finished! -----")
    logger.info(f"Number of training and validation images (across all sites): {train_ctr}")
    logger.info(f"Number of test images: {test_ctr}")
    # Get number of test images per site (site-003, site-012, etc.)
    test_images_per_site = {}
    for test_subject in test_subjects:
        site = find_site_in_path(test_subject)
        if site in test_images_per_site:
            test_images_per_site[site] += 1
        else:
            test_images_per_site[site] = 1
    # Print number of test images per site
    for site, num_images in test_images_per_site.items():
        logger.info(f"Number of test images in {site}: {num_images}")

    # create the yaml file containing the train and test niftis
    create_yaml(train_niftis, test_nifitis, path_out, args, train_ctr, test_ctr, dataset_commits)


if __name__ == "__main__":
    main()
