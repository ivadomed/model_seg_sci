import argparse
import os
from tqdm import tqdm
from collections import Counter

import pandas as pd
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from ivadomed.transforms import Resample, CenterCrop, \
     RandomAffine, ElasticTransform, NormalizeInstance, RandomReverse

# Argument parsing
parser = argparse.ArgumentParser(description='Visualization of Transformed Input Volumes')
parser.add_argument('-pd', '--path_data', type=str, required=True,
                    help='Path to the root folder containing the preprocessed dataset')
parser.add_argument('-po', '--path_output', type=str, required=True,
                    help="Path to the output folder")
args = parser.parse_args()

if not os.path.exists(args.path_output):
    os.makedirs(args.path_output)

# Values for Data augmentation methods
center_crop_size = [96, 384, 32]
hspace, wspace, dspace = 1.0, 1.0, 2.0

# Quick checking of arguments
if not os.path.exists(args.path_data):
    raise NotADirectoryError('%s could NOT be found!' % args.path_data)

# Get all subjects
subjects_df = pd.read_csv(os.path.join(args.path_data, 'participants.tsv'), sep='\t')
subjects = subjects_df['participant_id'].values.tolist()

# Log resolutions and sizes for data exploration
resolutions, sizes, crop_sizes = [], [], []

# Use Data augmentation transformation on each subject
for subject in tqdm(subjects, desc='Iterating over Subjects'):
    
    # Another for loop for going through sessions
    temp_subject_path = os.path.join(args.path_data, subject)
    num_sessions_per_subject = sum(os.path.isdir(os.path.join(temp_subject_path, pth)) for pth in os.listdir(temp_subject_path))
    
    # for ses_idx in tqdm(range(1, num_sessions_per_subject+1), desc="Iterating over Sessions"):
    for ses_idx in range(1, num_sessions_per_subject+1):
        # Get paths with session numbers
        session = 'ses-0' + str(ses_idx)
        subject_images_path = os.path.join(args.path_data, subject, session, 'anat')
        subject_labels_path = os.path.join(args.path_data, 'derivatives', 'labels', subject, session, 'anat')

        # Read original subject image (i.e. 3D volume) to be used for training
        img_path = os.path.join(subject_images_path, '%s_%s_acq-sag_T2w.nii.gz' % (subject, session))
        sag_img = nib.load(img_path)
        hfac, wfac, dfac = sag_img.header.get_zooms()
        # Read original and cropped subject ground-truths (GT)
        gt_fpath = os.path.join(subject_labels_path, '%s_%s_acq-sag_T2w_lesion-manual.nii.gz' % (subject, session))
        sag_gt = nib.load(gt_fpath)

        # Convert to Numpy
        sag_img, sag_gt = sag_img.get_fdata(), sag_gt.get_fdata()
        # print("After numpy conversion"); print(sag_img.shape, sag_img[:4, 60, :4])

        # Resample
        hfactor, wfactor, dfactor  = hfac/hspace, wfac/wspace, dfac/dspace
        params_resample = (hfactor, wfactor, dfactor)
        sag_img = zoom(sag_img, zoom=params_resample, order=1)
        sag_gt = zoom(sag_gt, zoom=params_resample, order=1)
        # print("After resampling"); print(sag_img.shape, sag_img[:4, 60, :4])

        # Apply center-cropping
        center_crop = CenterCrop(size=center_crop_size)
        sag_img = center_crop(sample=sag_img, metadata={'crop_params': {}})[0]
        sag_gt = center_crop(sample=sag_gt, metadata={'crop_params': {}})[0]
        # print("After center cropping"); print(sag_img.shape, sag_img[:4, 60, :4])

        # Apply random affine
        random_affine = RandomAffine(degrees=10, translate=[0.05, 0.05, 0.05], scale=[0.1, 0.1, 0.1])
        sag_img, metadata = random_affine(sample=sag_img, metadata={})
        sag_gt, _ = random_affine(sample=sag_gt, metadata=metadata)
        # print("After affine"); print(sag_img.shape, sag_img[:4, 60, :4])

        # Normalize to zero mean and unit variance
        normalize_instance = NormalizeInstance()
        if sag_img.std() < 1e-5:
            sag_img = sag_img - sag_img.mean()
        else:
            sag_img, _ = normalize_instance(sample=sag_img, metadata={})
        # sag_gt, _ = normalize_instance(sample=sag_gt, metadata={})
        # print("After normalization"); print(sag_img.shape, sag_img[:4, 60, :4])

        # Save the transformed images and GTs as new NIfTI files
        # print(sag_img.dtype)    # float64
        sag_img_nib = nib.Nifti1Image(sag_img, affine=np.eye(4))
        sag_gt_nib = nib.Nifti1Image(sag_gt, affine=np.eye(4))
        
        # Create BIDS-style output folder for transformed data
        output_images_path = os.path.join(args.path_output, subject, session, 'anat')
        output_gts_path = os.path.join(args.path_output, 'derivatives', 'labels', subject, session, 'anat')
        if not os.path.exists(output_images_path):
            os.makedirs(output_images_path)
        if not os.path.exists(output_gts_path):
            os.makedirs(output_gts_path)

        nib.save(img=sag_img_nib, filename=os.path.join(output_images_path, '%s_%s_acq-sag_T2w_aug.nii.gz' % (subject, session)))
        nib.save(img=sag_gt_nib, filename=os.path.join(output_gts_path, '%s_%s_acq-sag_T2w_lesion-manual_aug.nii.gz' % (subject, session)))

        # # Get and log size and resolution for each subject image
        # size = sag_img.get_fdata().shape
        # sizes.append(size)
        # # crop_sizes.append(crop_size)
        # # resolutions.append(resolution)
