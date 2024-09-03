"""
Convert BIDS-structured SCI datasets (site_007, site_012, etc.) to the nnUNetv2 structure.
REGION-BASED (default) and MULTICHANNEL training format depending on the input arguments.

REGION-BASED:
    - 1 input channel: T2w image
    - labels: lesions are considered as multi-class masks: `1` for hyperintense edema, `2` for hypointense hemorrhage.

MULTICHANNEL:
    - 2 input channels: T2w image and hyperintense edema segmentation (binary mask)
    - labels: binary mask for the hypointense hemorrhage

NOTE: the script performs RPI reorientation of the images and labels using SCT's Image class.

dataset.json:

```json
    "channel_names": {
        "0": "acq-sag_T2w"
    },
    "labels": {
        "background": 0,
        "lesion": [
            1,
            2
        ],
        "hemorrhage": 2
    },
    "regions_class_order": [
        1,
        2
    ],
```

Full details about the format can be found here:
https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

Usage example multiple datasets:
    python convert_bids_to_nnUNetv2_region-based.py
        --path-data ~/data/site_007 ~/data/site_012
        --path-out ${nnUNet_raw}
        -dname SCIlesions_hemorrhage
        -dnum 601
        --split 0.8 0.2
        --seed 50
        --region-based

Usage example single dataset:
    python convert_bids_to_nnUNetv2_region-based.py
        --path-data ~/data/site_007
        --path-out ${nnUNet_raw}
        -dname SCIlesions_hemorrhage
        -dnum 601
        --split 0.8 0.2
        --seed 50
        --region-based

You can run the script using the SCT's conda venv:
    source ./python/etc/profile.d/conda.sh
    conda activate venv_sct


Authors: Jan Valosek, Naga Karthik
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
from utils import binarize_label, create_region_based_label, get_git_branch_and_commit, Image, create_multi_channel_label_input
from tqdm import tqdm
import nibabel as nib


LABEL_SUFFIXES = {
    "site_007": ["lesion", "lesion_hemorrhage"]
}


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Convert BIDS-structured dataset to nnUNetV2 format.')
    parser.add_argument('--path-data', nargs='+', required=True, type=str,
                        help='Path to BIDS dataset(s) (list).')
    parser.add_argument('--path-out', help='Path to output directory.', required=True)
    parser.add_argument('--dataset-name', '-dname', default='SCIlesions_hemorrhage', type=str,
                        help='Specify the task name. It will be combined with the "--dataset-number" into '
                             '"Dataset<dataset_number>_<dataset_name>"')
    parser.add_argument('--dataset-number', '-dnum', default=601, type=int,
                        help='Specify the task number, has to be greater than 500 but less than 999. e.g 502.'
                             'It will be combined with the "--dataset-name" into '
                             '"Dataset<dataset_number>_<dataset_name>"')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to be used for the random number generator split into training and test sets.')
    parser.add_argument('--region-based', action='store_true', default=False,
                        help='If set, the script will create labels for region-based nnUNet training. The script will '
                             'automatically include "MultiChannel" to the dataset name. '
                             'Default: True')
    parser.add_argument('--multichannel', action='store_true', default=False,
                        help='If set, the script will create multi-channel input (edema + hemorrhage) nnUNet training. '
                             'The script will automatically include "MultiChannel" to the dataset name. '
                             'Default: False')
    parser.add_argument('--crop', action='store_true', default=False,
                        help='If set, the script will crop the image and labels based on the edema mask. '
                             'The script will automatically include "MultiChannel" to the dataset name. '
                             'Default: False')
    # argument that accepts a list of floats as train val test splits
    parser.add_argument('--split', nargs='+', type=float, default=[0.8, 0.2],
                        help='Ratios of training (includes validation) and test splits lying between 0-1. Example: '
                             '--split 0.8 0.2')
    return parser


def get_region_based_label(subject_label_file, subject_image_file, site_name, sub_ses_name, thr=0.5):
    # define path for sc seg file
    subject_seg_file = subject_label_file.replace(f'_{LABEL_SUFFIXES[site_name][1]}', f'_{LABEL_SUFFIXES[site_name][0]}')

    # check if the seg file exists
    if not os.path.exists(subject_seg_file):
        logger.info(f"Spinal cord segmentation file for subject {sub_ses_name} does not exist. Skipping.")
        return None

    # create region-based label
    seg_lesion_nii = create_region_based_label(subject_label_file, subject_seg_file, subject_image_file,
                                               sub_ses_name, thr=thr)

    # save the region-based label
    combined_seg_file = subject_label_file.replace(f'_{LABEL_SUFFIXES[site_name][1]}', '_sc-lesion')
    nib.save(seg_lesion_nii, combined_seg_file)

    return combined_seg_file


def get_multi_channel_label_input(subject_label_file, subject_image_file, site_name, sub_ses_name, thr=0.5):
    # define path for sc seg file
    subject_seg_file = subject_label_file.replace(f'_{LABEL_SUFFIXES[site_name][1]}', f'_{LABEL_SUFFIXES[site_name][0]}')

    # check if the seg file exists
    if not os.path.exists(subject_seg_file):
        logger.info(f"Spinal cord segmentation file for subject {sub_ses_name} does not exist. Skipping.")
        return None

    # create label for the multi-channel training
    # Here, the label is the edema seg that will be the 2nd channel in the input along with the image
    # we ensure that the lesion seg is part of the spinal cord seg
    seg_lesion_nii = create_multi_channel_label_input(subject_label_file, subject_seg_file, subject_image_file,
                                                      sub_ses_name, thr=thr)

    # save the region-based label
    combined_seg_file = subject_label_file.replace(f'_{LABEL_SUFFIXES[site_name][1]}', '_sc')
    nib.save(seg_lesion_nii, combined_seg_file)

    return combined_seg_file


def create_directories(path_out, site):
    """Create test directories for a specified site.

    Args:
    path_out (str): Base output directory.
    site (str): Site identifier, such as 'dcm-zurich-lesions
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
    # Find 'dcm-zurich-lesions' or 'dcm-zurich-lesions-20231115'
    if 'dcm' in path:
        match = re.search(r'dcm-zurich-lesions(-\d{8})?', path)
    elif 'sci' in path:
        match = re.search(r'sci-zurich|sci-colorado|sci-paris', path)
    elif 'site' in path:
        # NOTE: PRAXIS data has 'site-xxx' or 'site_xxx' in the path
        # (and doesn't have the site names themselves in the path)
        match = re.search(r'site-\d{3}|site_\d{3}', path)

    return match.group(0) if match else None


def create_yaml(train_niftis, test_nifitis, path_out, args):
    """
    Create a yaml file containing the list of training and test images.
    :param train_niftis:
    :param test_nifitis:
    :param path_out:
    :param args:
    :return:
    """
    niftis_dict = {
        f"train": sorted(train_niftis),
        f"test": sorted(test_nifitis)
    }

    # write the train and test niftis to a yaml file
    with open(os.path.join(path_out, f"train_test_split_seed{args.seed}.yaml"), "w") as outfile:
        yaml.dump(niftis_dict, outfile, default_flow_style=False)


def create_dataset_json(args, train_ctr, test_ctr, dataset_commits, path_out):
    """
    Create the dataset.json file for the nnUNet dataset.
    :param args:
    :param train_ctr:
    :param test_ctr:
    :param dataset_commits:
    :param path_out:
    :return:
    """
    json_dict = OrderedDict()
    json_dict['name'] = args.dataset_name
    json_dict['description'] = args.dataset_name
    json_dict['numTraining'] = train_ctr
    json_dict['numTest'] = test_ctr
    json_dict['seed_used'] = args.seed
    json_dict['dataset_versions'] = dataset_commits
    json_dict['image_orientation'] = "RPI"

    # The following keys are the most important ones.
    # REGION-BASED
    if args.region_based:
        json_dict['channel_names'] = {
            0: "T2w",
        }

        json_dict['labels'] = {
            "background": 0,
            "lesion": [1, 2],
            "hemorrhage": 2,
        }
        json_dict['regions_class_order'] = [1, 2]
    # MULTI-CHANNEL
    elif args.multichannel:
        json_dict['channel_names'] = {
            0: "T2w",
            1: "lesion",
        }

        json_dict['labels'] = {
            "background": 0,
            "lesion": 1,
        }

    else:
        json_dict['channel_names'] = {
            0: "T2w",
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


def main():
    parser = get_parser()
    args = parser.parse_args()

    train_ratio, test_ratio = args.split
    # Update dataset name with the type of training (region-based or multi-channel)
    if args.region_based:
        path_out = Path(os.path.join(os.path.abspath(args.path_out),
                                     f'Dataset{args.dataset_number}_{args.dataset_name}_RegionBasedSeed{args.seed}'))
    elif args.multichannel:
        path_out = Path(os.path.join(os.path.abspath(args.path_out),
                                     f'Dataset{args.dataset_number}_{args.dataset_name}_MultiChannelSeed{args.seed}'))

    # replace 'Seed' by 'CropSeed' if crop is set
    if args.crop:
        path_out = Path(str(path_out).replace('Seed', 'CropSeed'))

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

    all_lesion_files, train_images, test_images = [], {}, {}
    # temp dict for storing dataset commits
    dataset_commits = {}

    # loop over the datasets
    for dataset in args.path_data:
        root = Path(dataset)

        # get the git branch and commit ID of the dataset
        dataset_name = os.path.basename(os.path.normpath(dataset))
        branch, commit = get_git_branch_and_commit(dataset)
        dataset_commits[dataset_name] = f"git-{branch}-{commit}"
        site_name = find_site_in_path(dataset)

        # get recursively all GT '_label-lesion' files
        hemorrhage_label_suffix = LABEL_SUFFIXES[site_name][1]
        hemorrhage_files = [str(path) for path in root.rglob(f'*_{hemorrhage_label_suffix}.nii.gz')]
        # add to the list of all subjects
        all_lesion_files.extend(hemorrhage_files)

        # Get the training and test splits
        # NOTE: we need a patient-wise split (not image-wise split) to ensure that the same patient with multiple
        # sessions is not present in both training and test sets
        subs = sorted([sub for sub in os.listdir(os.path.join(root, 'derivatives', 'labels'))])
        tr_subs, te_subs = train_test_split(subs, test_size=test_ratio, random_state=args.seed)

        for sub in tr_subs:
            # get the lesion files for the subject
            lesion_files_sub = [file for file in hemorrhage_files if sub in file]

            for lesion_file in lesion_files_sub:
                if not os.path.exists(lesion_file):
                    logger.info(f"Lesion file {lesion_file} does not exist. Skipping.")
                    continue
                # add the lesion file to the training set
                train_images[lesion_file] = lesion_file

        for sub in te_subs:
            # get the lesion files for the subject
            lesion_files_sub = [file for file in hemorrhage_files if sub in file]

            for lesion_file in lesion_files_sub:
                if not os.path.exists(lesion_file):
                    logger.info(f"Lesion file {lesion_file} does not exist. Skipping.")
                    continue
                # add the lesion file to the test set
                test_images[lesion_file] = lesion_file

    logger.info(f"Found subjects in the training set (combining all datasets): {len(train_images)}")
    logger.info(f"Found subjects in the test set (combining all datasets): {len(test_images)}")
    # Print test images for each site
    for site in sites:
        logger.info(f"Test subjects in {site}: {len([sub for sub in test_images if site in find_site_in_path(sub)])}")

    # print version of each dataset in a separate line
    for dataset_name, dataset_commit in dataset_commits.items():
        logger.info(f"{dataset_name} dataset version: {dataset_commit}")

    # Counters for train and test sets
    train_ctr, test_ctr = 0, 0
    train_niftis, test_nifitis = [], []
    # Loop over all images
    for subject_label_file in tqdm(all_lesion_files, desc="Iterating over all images"):

        site_name = find_site_in_path(subject_label_file)
        # Construct path to the background image, i.e., remove 'derivatives/labels' and the suffix
        subject_image_file = subject_label_file.replace('/derivatives/labels', '').replace(f'_{LABEL_SUFFIXES[site_name][1]}', '')

        # Train images
        if subject_label_file in train_images.keys():

            train_ctr += 1
            # add the subject image file to the list of training niftis
            train_niftis.append(os.path.basename(subject_image_file))

            # create the new convention names for nnunet
            sub_name = f"{str(Path(subject_image_file).name).replace('.nii.gz', '')}"

            subject_image_file_nnunet = os.path.join(path_out_imagesTr,
                                                     f"{args.dataset_name}_{site_name}_{sub_name}_{train_ctr:03d}_0000.nii.gz")
            subject_label_file_nnunet = os.path.join(path_out_labelsTr,
                                                     f"{args.dataset_name}_{site_name}_{sub_name}_{train_ctr:03d}.nii.gz")

            # MULTI-CHANNEL
            if args.multichannel:
                if args.region_based:
                    raise ValueError("Multi-channel input is not supported with region-based labels.")

                # Create channel 1 file (with 0001 suffix). Just a reminder:
                # channel 0: image (0000)
                # channel 1: edema (0001)
                subject_edema_file_nnunet = os.path.join(path_out_imagesTr,
                                                      f"{args.dataset_name}_{site_name}_{sub_name}_{train_ctr:03d}_0001.nii.gz")

                # overwritten the subject_edema_file_nnunet with the label for multi-channel training
                # (hemorrhage is part of edema)
                subject_edema_file = get_multi_channel_label_input(subject_label_file, subject_image_file,
                                                                   site_name, sub_name, thr=0.5)

                if subject_edema_file is None:
                    print(f"Skipping since the multi-channel label could not be generated")
                    continue

            # REGION-BASED
            elif args.region_based:
                # overwritten the subject_label_file with the region-based label
                subject_label_file = get_region_based_label(subject_label_file, subject_image_file,
                                                            site_name, sub_name, thr=0.5)
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

            # Copy the edema file (channel 0001) for multi-channel training
            if args.multichannel:
                shutil.copyfile(subject_edema_file, subject_edema_file_nnunet)
                # convert the edema seg to RPI using the Image class
                edema_image = Image(subject_edema_file_nnunet)
                edema_image.change_orientation("RPI")
                edema_image.save(subject_edema_file_nnunet)
                if args.crop:
                    # Not implemented yet, TODO
                    pass

            # don't binarize the label if either of the region-based or multi-channel training is set
            if not args.region_based:
                binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)

            # Crop image and labels based on the edema mask
            if args.crop:
                # Use sct_crop_image
                # Note: since the T2w images are sagittal with thick sagittal slices (usually ~3mm), we only crop by
                # 5 pixels in R-L (x-axis)
                os.system(f'sct_crop_image -i {subject_image_file_nnunet} -m {subject_label_file_nnunet} -dilate 5x10x10 -o {subject_image_file_nnunet}')
                os.system(f'sct_crop_image -i {subject_label_file_nnunet} -m {subject_label_file_nnunet} -dilate 5x10x10 -o {subject_label_file_nnunet}')

        # Test images
        elif subject_label_file in test_images:

            test_ctr += 1
            # add the image file to the list of testing niftis
            test_nifitis.append(os.path.basename(subject_image_file))

            # create the new convention names for nnunet
            sub_name = f"{str(Path(subject_image_file).name).replace('.nii.gz', '')}"

            subject_image_file_nnunet = os.path.join(Path(path_out,
                                                          f'imagesTs_{find_site_in_path(test_images[subject_label_file])}'),
                                                     f'{args.dataset_name}_{site_name}_{sub_name}_{test_ctr:03d}_0000.nii.gz')
            subject_label_file_nnunet = os.path.join(Path(path_out,
                                                          f'labelsTs_{find_site_in_path(test_images[subject_label_file])}'),
                                                     f'{args.dataset_name}_{site_name}_{sub_name}_{test_ctr:03d}.nii.gz')

            # MULTI-CHANNEL
            if args.multichannel:
                if args.region_based:
                    raise ValueError("Multi-channel input is not supported with region-based labels.")

                # channel 0: image, channel 1: edema seg
                subject_edema_file_nnunet = os.path.join(Path(path_out,
                                                          f'imagesTs_{find_site_in_path(test_images[subject_label_file])}'),
                                                     f'{args.dataset_name}_{site_name}_{sub_name}_{test_ctr:03d}_0001.nii.gz')

                # overwritten the subject_edema_file_nnunet with the label for multi-channel training
                # (hemorrhage is part of edema)
                subject_edema_file = get_multi_channel_label_input(subject_label_file, subject_image_file,
                                                                   site_name, sub_name, thr=0.5)

                if subject_edema_file is None:
                    print(f"Skipping since the multi-channel label could not be generated")
                    continue

            # REGION-BASED
            elif args.region_based:
                # overwritten the subject_label_file with the region-based label
                subject_label_file = get_region_based_label(subject_label_file, subject_image_file,
                                                            site_name, sub_name, thr=0.5)
                if subject_label_file is None:
                    continue

            # copy the files to new structure
            shutil.copyfile(subject_image_file, subject_image_file_nnunet)
            shutil.copyfile(subject_label_file, subject_label_file_nnunet)
            # print(f"\nCopying {subject_image_file} to {subject_image_file_nnunet}")
            # convert the image and label to RPI using the Image class
            image = Image(subject_image_file_nnunet)
            image.change_orientation("RPI")
            image.save(subject_image_file_nnunet)

            label = Image(subject_label_file_nnunet)
            label.change_orientation("RPI")
            label.save(subject_label_file_nnunet)

            if args.multichannel:
                shutil.copyfile(subject_edema_file, subject_edema_file_nnunet)
                # convert the edema seg to RPI using the Image class
                edema_image = Image(subject_edema_file_nnunet)
                edema_image.change_orientation("RPI")
                edema_image.save(subject_edema_file_nnunet)
                if args.crop:
                    # Not implemented yet, TODO
                    pass

            # don't binarize the label if either of the region-based or multi-channel training is set
            if not args.region_based:
                binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)

            # Crop image and labels based on the edema mask
            if args.crop:
                # Use sct_crop_image
                # Note: since the T2w images are sagittal with thick sagittal slices (usually ~3mm), we only crop by
                # 5 pixels in R-L (x-axis)
                os.system(f'sct_crop_image -i {subject_image_file_nnunet} -m {subject_label_file_nnunet} -dilate 5x10x10 -o {subject_image_file_nnunet}')
                os.system(f'sct_crop_image -i {subject_label_file_nnunet} -m {subject_label_file_nnunet} -dilate 5x10x10 -o {subject_label_file_nnunet}')

        else:
            print("Skipping file, could not be located in the Train or Test splits split.", subject_label_file)

    logger.info(f"----- Dataset conversion finished! -----")
    logger.info(f"Number of training and validation images (across all sites): {train_ctr}")
    # Get number of train and val images per site
    train_images_per_site = {}
    for train_subject in train_images:
        site = find_site_in_path(train_subject)
        if site in train_images_per_site:
            train_images_per_site[site] += 1
        else:
            train_images_per_site[site] = 1
    # Print number of train images per site
    for site, num_images in train_images_per_site.items():
        logger.info(f"Number of training and validation images in {site}: {num_images}")

    logger.info(f"Number of test images (across all sites): {test_ctr}")
    # Get number of test images per site
    test_images_per_site = {}
    for test_subject in test_images:
        site = find_site_in_path(test_subject)
        if site in test_images_per_site:
            test_images_per_site[site] += 1
        else:
            test_images_per_site[site] = 1
    # Print number of test images per site
    for site, num_images in test_images_per_site.items():
        logger.info(f"Number of test images in {site}: {num_images}")

    # create the yaml file containing the train and test images
    create_yaml(train_niftis, test_nifitis, path_out, args)
    # create the dataset.json file
    create_dataset_json(args, train_ctr, test_ctr, dataset_commits, path_out)


if __name__ == "__main__":
    main()
