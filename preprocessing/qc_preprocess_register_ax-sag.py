"""
Quality control for preprocessing step.
See `preprocess_data.sh` for the preprocessing pipeline.
Adapted to process longitudinal data from: 
https://github.com/ivadomed/model_seg_ms_mp2rage/blob/main/preprocessing/qc_preprocess.py
"""

import argparse
import os
from tqdm import tqdm
from collections import Counter

import pandas as pd
import nibabel as nib
import numpy as np

# Argument parsing
parser = argparse.ArgumentParser(description='Quality control for preprocessing.')
parser.add_argument('-s', '--sct_output_path', type=str, required=True,
                    help='Path to the folder generated by `sct_run_batch`. This folder should contain `data_processed` folder.')
args = parser.parse_args()

# Quick checking of arguments
if not os.path.exists(args.sct_output_path):
    raise NotADirectoryError('%s could NOT be found!' % args.sct_output_path)
else:
    if not os.path.exists(os.path.join(args.sct_output_path, 'data_processed')):
        raise NotADirectoryError('`data_processed` could NOT be found within %s' % args.sct_output_path)

# Get all subjects
subjects_df = pd.read_csv(os.path.join(args.sct_output_path, 'data_processed', 'participants.tsv'), sep='\t')
subjects = subjects_df['participant_id'].values.tolist()

# Log resolutions and sizes for data exploration
resolutions, sizes, crop_sizes, axial_crop_sizes = [], [], [], []

# Log problematic subjects for QC
failed_crop_subjects, shape_mismatch_subjects, left_out_lesion_subjects = [], [], []

# Perform QC on each subject
for subject in tqdm(subjects, desc='Iterating over Subjects'):
    
    # Another for loop for going through sessions
    temp_subject_path = os.path.join(args.sct_output_path, 'data_processed', subject)
    num_sessions_per_subject = sum(os.path.isdir(os.path.join(temp_subject_path, pth)) for pth in os.listdir(temp_subject_path))
    
    for ses_idx in tqdm(range(1, num_sessions_per_subject+1), desc="Iterating over Sessions"):
        # Get paths with session numbers
        session = 'ses-0' + str(ses_idx)
        subject_images_path = os.path.join(args.sct_output_path, 'data_processed', subject, session, 'anat')
        subject_labels_path = os.path.join(args.sct_output_path, 'data_processed', 'derivatives', 'labels', subject, session, 'anat')

        # Read original and cropped subject image (i.e. 3D volume) to be used for training
        img_path = os.path.join(subject_images_path, '%s_%s_acq-sag_T2w.nii.gz' % (subject, session))
        img_crop_res_fpath = os.path.join(subject_images_path, '%s_%s_acq-sag_T2w_crop_res.nii.gz' % (subject, session))
        ax_img_reg_fpath = os.path.join(subject_images_path, '%s_%s_acq-ax_T2w_reg.nii.gz' % (subject, session))
        if not os.path.exists(img_crop_res_fpath):
            failed_crop_subjects.append([subject, session])
            continue
        img = nib.load(img_path)
        img_crop_res = nib.load(img_crop_res_fpath)
        ax_img_reg = nib.load(ax_img_reg_fpath)

        # Get and log size and resolution for each subject image
        size = img.get_fdata().shape
        crop_size = img_crop_res.get_fdata().shape
        ax_crop_size = ax_img_reg.get_fdata().shape
        resolution = tuple(img_crop_res.header['pixdim'].tolist()[1:4])
        resolution = tuple([np.round(r, 1) for r in list(resolution)])
        sizes.append(size)
        crop_sizes.append(crop_size)
        axial_crop_sizes.append(ax_crop_size)
        resolutions.append(resolution)

        # Read original and cropped subject ground-truths (GT)
        gt_fpath = os.path.join(subject_labels_path, '%s_%s_acq-sag_T2w_lesion-manual.nii.gz' % (subject, session))
        gt_crop_fpath = os.path.join(subject_labels_path, '%s_%s_acq-sag_T2w_lesion-manual_crop.nii.gz' % (subject, session))
        gt_crop_res_fpath = os.path.join(subject_labels_path, '%s_%s_acq-sag_T2w_lesion-manual_crop_res.nii.gz' % (subject, session))

        gt = nib.load(gt_fpath)
        gt_crop = nib.load(gt_crop_fpath)
        gt_crop_res = nib.load(gt_crop_res_fpath)

        # Basic shape checks
        if not img_crop_res.shape == gt_crop_res.shape:
            shape_mismatch_subjects.append([subject, session])
            continue

        # Check if the dilated SC mask leaves out any lesions from GTs
        # comparing the original and (only) the cropped GT because the subsequent resampling will not crop lesions
        if not (np.allclose(np.sum(gt.get_fdata()), np.sum(gt_crop.get_fdata()))):
            left_out_lesion_subjects.append([subject, session])

print('RESOLUTIONS: ', Counter(resolutions), "\n")
print('SIZES: ', Counter(sizes), "\n")
print('CROP SIZES: ', Counter(crop_sizes), "\n")
print('AXIAL CROP SIZES: ', Counter(axial_crop_sizes), "\n")


print('Could not find cropped image for the following subjects: ', failed_crop_subjects)
print('Found shape mismatch in images and GTs for the following subjects: ', shape_mismatch_subjects)
print('ALERT: Lesion(s) from raters cropped during preprocessing for the following subjects: ', left_out_lesion_subjects)