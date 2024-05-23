"""
Convert BIDS-structured SCI datasets (sci-zurich, sci-colorado, dcm-zurich-lesions, dcm-zurich-lesions-20231115, etc.) to the nnUNetv2 
REGION-BASED and MULTICHANNEL training format depending on the input arguments.

dataset.json:

```json
    "channel_names": {
        "0": "acq-ax_T2w"
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

The script in default creates region-based labels for segmenting both lesion and the spinal cord.

Currently only supports the conversion of a single contrast. In case of multiple contrasts, the script should be
modified to include those as well.

Note: the script performs RPI reorientation of the images and labels

Usage example multiple datasets:
    python convert_bids_to_nnUNetv2_region-based.py
        --path-data ~/data/dcm-zurich-lesions ~/data/dcm-zurich-lesions-20231115
        --path-out ${nnUNet_raw}
        -dname DCMlesions
        -dnum 601
        --split 0.8 0.2
        --seed 50
        --region-based

Usage example single dataset:
    python convert_bids_to_nnUNetv2_region-based.py
        --path-data ~/data/dcm-zurich-lesions
        --path-out ${nnUNet_raw}
        -dname DCMlesions
        -dnum 601
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


LABEL_SUFFIXES = {
    "dcm-zurich-lesions": ["label-SC_mask-manual", "label-lesion"],
    "dcm-zurich-lesions-20231115": ["label-SC_mask-manual", "label-lesion"],
    "sci-colorado": ["seg-manual", "lesion-manual"],
    "sci-paris": ["seg-manual", "lesion-manual"],
    "sci-zurich": ["seg-manual", "lesion-manual"],
    "site-003": ["seg", "lesion"],
    "site-012": ["seg", "lesion"],
}
# NOTE: these datasets only contain a few subjects (n<20), hence using them all for training
TRAIN_ONLY_SITES = ['dcm-zurich-lesions', 'sci-paris', 'site-012']
TEST_ONLY_SITES = ['site-003']


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
    # input yaml file containing list of axial subjects to include for active learning
    parser.add_argument('--include-axial', type=str, default=None,
                        help='Path to yaml file containing list of axial subjects to include for active learning.')
    parser.add_argument("--add-sci-paris", action="store_true", default=False,
                        help="If set, the script will add the sci-paris dataset to the training set. Default: False")

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
def get_multi_channel_label_input(subject_label_file, subject_image_file, site_name, sub_ses_name, thr=0.5):
    # define path for sc seg file
    subject_seg_file = subject_label_file.replace(f'_{LABEL_SUFFIXES[site_name][1]}', f'_{LABEL_SUFFIXES[site_name][0]}')

    # check if the seg file exists
    if not os.path.exists(subject_seg_file):
        logger.info(f"Spinal cord segmentation file for subject {sub_ses_name} does not exist. Skipping.")
        return None

    # create label for the multi-channel training. Here, the label is the SC seg that will be the 2nd channel in the input
    # along with the image. (we ensure that the lesion seg is part of the spinal cord seg
    seg_lesion_nii = create_multi_channel_label_input(subject_label_file, subject_seg_file, subject_image_file, 
                                                      sub_ses_name, thr=thr)

    # save the region-based label
    combined_seg_file = subject_label_file.replace(f'_{LABEL_SUFFIXES[site_name][1]}', '_sc')
    nib.save(seg_lesion_nii, combined_seg_file)

    return combined_seg_file

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
    json_dict['numTest'] = test_ctr_zur + test_ctr_col
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


if __name__ == "__main__":
    main()
