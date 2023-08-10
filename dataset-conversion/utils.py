import nibabel as nib
import numpy as np


def binarize_label(subject_path, label_path):
    label_npy = nib.load(label_path).get_fdata()
    threshold = 0.5
    label_npy = np.where(label_npy > threshold, 1, 0)
    ref = nib.load(subject_path)
    label_bin = nib.Nifti1Image(label_npy, ref.affine, ref.header)
    # overwrite the original label file with the binarized version
    nib.save(label_bin, label_path)


def create_region_based_label(lesion_label_file, seg_label_file, image_file, sub_ses_name, thr=0.5):
    """
    Creates region-based labels for nnUNet training. The regions are: 
    0: background
    1: spinal cord seg
    2: lesion seg
    """
    # load the labels
    lesion_label_npy = nib.load(lesion_label_file).get_fdata()
    seg_label_npy = nib.load(seg_label_file).get_fdata()

    # binarize the labels
    lesion_label_npy = np.where(lesion_label_npy > thr, 1, 0)
    seg_label_npy = np.where(seg_label_npy > thr, 1, 0)

    # check if the shapes of the labels match
    assert lesion_label_npy.shape == seg_label_npy.shape, \
          f'Shape mismatch between lesion label and segmentation label for subject {sub_ses_name}. Check the labels.'

    # create a new label array with the same shape as the original labels
    label_npy = np.zeros(lesion_label_npy.shape, dtype=np.int16)
    # spinal cord
    label_npy[seg_label_npy == 1] = 1
    # lesion seg
    label_npy[lesion_label_npy == 1] = 2
    # TODO: what happens when the subject has no lesion?

    # print unique values in the label array
    # print(f'Unique values in the label array for subject {sub_ses_name}: {np.unique(label_npy)}')
    
    # save the new label file
    ref = nib.load(image_file)
    label_nii = nib.Nifti1Image(label_npy, ref.affine, ref.header)

    return label_nii
