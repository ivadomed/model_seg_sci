"""
Take lesion from subject_b and insert it into subject_a

Activate SCT conda environment:
    source ${SCT_DIR}/python/etc/profile.d/conda.sh
    conda activate venv_sct

Run:
    python generate_new_lesion.py

nnUNet data structure is required.
TODO: switch to BIDS?
"""
import sys
import time
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import argparse
import os

from spinalcordtoolbox.image import Image, zeros_like
from spinalcordtoolbox.resampling import resample_nib

# TODO: Check out Diffusion models for synthesizing new images + lesions 


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-num", default=100, type=int, help="Total number of newly generated subjects. Default: 100")
    parser.add_argument("-dir-pathology", default="imagesTr", type=str,
                        help="Path to raw images from pathology dataset (i.e. SCI-Zurich)")
    parser.add_argument("-dir-lesions", default="labelsTr", type=str,
                        help="Path to lesion labels from pathology dataset (i.e. SCI-Zurich)")
    parser.add_argument("-dir-masks-pathology", default="masksTr", type=str,
                        help="Path to SC masks from pathology dataset (i.e. SCI-Zurich)")
    parser.add_argument("-dir-healthy", default="imagesTr", type=str,
                        help="Path to raw images from the healthy dataset (i.e. Spine Generic Multi)")
    parser.add_argument("-dir-masks-healthy", default="masksTr", type=str,
                        help="Path to SC masks from healthy dataset (i.e. Spine Generic Multi)")
    parser.add_argument("-dir-save", default="labelsTr", type=str,
                        help="Path to save new lesion samples")
    parser.add_argument("-seed", default=99, type=int, help="Random seed used for subject mixing. Default: 99")
    parser.add_argument("-resample", default=False, action='store_true',
                        help="Resample the augmented images to the resolution of pathological dataset. Default: False")
    parser.add_argument("-qc", default=False, action='store_true', help="Perform QC using sct_qc. Default: False")
    parser.add_argument("-min-lesion-vol", "--min-lesion-volume", default=400, type=float,
                        help="Minimum lesion volume in mm^3. Default: 400")
    # parser.add_argument("--mask_save_path", "-mask-pth", default="mask", type=str,
    #                     help="Path to save carved masks")

    return parser


def coefficient_of_variation(masked_image):
    return np.std(masked_image, ddof=1) / np.mean(masked_image) * 100


def get_centerline(im_healthy_sc_data):
    # Get centerline of the healthy SC
    # for each slice in the mask_sc, get the center coordinate of the z-axis
    num_z_slices = im_healthy_sc_data.shape[2]
    healthy_centerline = list()
    for z in range(num_z_slices):
        x, y = ndimage.center_of_mass(im_healthy_sc_data[:, :, z])
        # check if not nan (because spinal cord mask covers only spinal cord, not the whole image and all slices)
        if not np.isnan(x) and not np.isnan(y):
            healthy_centerline.append((round(x), round(y), z))

    return healthy_centerline


def get_lesion_volume(im_patho_lesion_data, voxel_dims):
    # Compute volume
    nonzero_voxel_count = np.count_nonzero(im_patho_lesion_data)
    voxel_volume = np.prod(voxel_dims)
    nonzero_voxel_volume = nonzero_voxel_count * voxel_volume

    # print("Number of non-zero voxels = {}".format(nonzero_voxel_count))
    print(f"Volume of non-zero voxels = {nonzero_voxel_volume:.2f} mm^3")

    return nonzero_voxel_volume


def insert_lesion(new_target, new_lesion, im_patho_data, im_patho_lesion_data, im_healthy_sc_data, coords,
                  new_position, intensity_ratio):
    """"
    Insert lesion from the bounding box to the new_target
    """
    # Get bounding box coordinates
    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0) + 1  # slices are exclusive at the top

    # Get coordinates where to insert the lesion
    x, y, z = new_position

    # TODO - take angle of the centerline into account when projecting the lesion
    # TODO for Nathan - rewrite this without 3 loops

    for x_step, x_cor in enumerate(range(x0, x1)):
        for y_step, y_cor in enumerate(range(y0, y1)):
            for z_step, z_cor in enumerate(range(z0, z1)):
                # Check that dimensions do not overflow
                if x + x_step >= new_target.shape[0] or y + y_step >= new_target.shape[1] or z + z_step >= new_target.shape[2]:
                    continue
                # Insert only voxels corresponding to the lesion mask (label_b)
                # Also make sure that the new lesion is not projected outside of the SC
                if im_patho_lesion_data[x_cor, y_cor, z_cor] > 0 and im_healthy_sc_data[x + x_step, y + y_step, z + z_step] > 0:
                    new_target[x + x_step, y + y_step, z + z_step] = im_patho_data[x_cor, y_cor, z_cor] * intensity_ratio
                    new_lesion[x + x_step, y + y_step, z + z_step] = im_patho_lesion_data[x_cor, y_cor, z_cor]

    return new_target, new_lesion


def generate_new_sample(sub_healthy, sub_patho, args, index):

    """
    Load healthy subject image and spinal cord segmentation
    """
    # Construct paths
    path_image_healthy = os.path.join(args.dir_healthy, sub_healthy + '_0000.nii.gz')
    path_mask_sc_healthy = os.path.join(args.dir_masks_healthy, sub_healthy + '.nii.gz')

    # Load image_healthy and mask_sc
    im_healthy = Image(path_image_healthy)
    im_healthy_sc = Image(path_mask_sc_healthy)

    im_healthy_orientation_native = im_healthy.orientation
    print(f"Healthy subject {path_image_healthy}: {im_healthy_orientation_native, im_healthy.dim[4:7]}")

    # Reorient to RPI
    im_healthy.change_orientation("RPI")
    im_healthy_sc.change_orientation("RPI")
    print("Reoriented to RPI")

    # Get numpy arrays
    im_healthy_data = im_healthy.data
    im_healthy_sc_data = im_healthy_sc.data

    # Check if image and spinal cord mask have the same shape, if not, skip this subject
    if im_healthy_data.shape != im_healthy_sc_data.shape:
        print("Warning: image_healthy and mask_sc have different shapes --> skipping subject")
        return

    """
    Load pathological subject image, spinal cord segmentation, and lesion mask
    """
    # Construct paths
    path_image_patho = os.path.join(args.dir_pathology, sub_patho + '_0000.nii.gz')
    path_label_patho = os.path.join(args.dir_lesions, sub_patho + '.nii.gz')
    path_mask_sc_patho = os.path.join(args.dir_masks_pathology, sub_patho + '.nii.gz')

    # Load image_patho, label_patho, and mask_sc_patho
    im_patho = Image(path_image_patho)
    im_patho_sc = Image(path_mask_sc_patho)
    im_patho_lesion = Image(path_label_patho)

    im_patho_orientation_native = im_patho.orientation
    print(f"Pathological subject {path_image_patho}: {im_patho_orientation_native, im_patho.dim[4:7]}")

    # Reorient to RPI
    im_patho.change_orientation("RPI")
    im_patho_sc.change_orientation("RPI")
    im_patho_lesion.change_orientation("RPI")
    print("Reoriented to RPI")

    # Get numpy arrays
    im_patho_data = im_patho.data
    im_patho_sc_data = im_patho_sc.data
    im_patho_lesion_data = im_patho_lesion.data

    # Check if image and spinal cord mask have the same shape, if not, skip this subject
    if im_patho_data.shape != im_patho_sc_data.shape:
        print("Warning: image_patho and label_patho have different shapes --> skipping subject\n")
        return

    # Check if lesion volume is less than X mm^3, if yes, skip this subject
    im_patho_lesion_vol = get_lesion_volume(im_patho_lesion_data, im_patho.dim[4:7])
    # TODO: set min_lesion_volume to 400 (?)
    if im_patho_lesion_vol < args.min_lesion_volume:
        print("Warning: lesion volume is too small --> skipping subject\n")
        return

    
    """
    Get intensity ratio between healthy and pathological SC and normalize images.
    The ratio is used to multiply the lesion in the healthy image.
    """
    # First, compute the difference in mean intensity between lesion and SC in the pathological image
    lesion_sc_diff = abs((np.mean(im_patho_data[im_patho_lesion_data > 0]) -
                         np.mean(im_patho_data[im_patho_sc_data > 0])) / np.mean(im_patho_data[im_patho_sc_data > 0]))

    print(f"Lesion/SC intensity difference (higher value means that the lesion is more hyperintense): {lesion_sc_diff}")

    # Second, get ratio healthy/patho SC
    intensity_ratio = coefficient_of_variation(im_healthy_data[im_healthy_sc_data > 0]) / \
                      coefficient_of_variation(im_patho_data[im_patho_sc_data > 0])

    # Finally, take into account the difference in intensity between lesion and patho SC
    intensity_ratio = intensity_ratio * lesion_sc_diff

    # Make sure the intensity ratio is always > 1 (i.e. the lesion is always brighter than the healthy SC)
    if intensity_ratio < 1:
        intensity_ratio = 1 / intensity_ratio

    print(f"Intensity ratio: {intensity_ratio}")

    # normalize images to range 0 and 1
    im_healthy_data = (im_healthy_data - np.min(im_healthy_data)) / (np.max(im_healthy_data) - np.min(im_healthy_data))
    im_patho_data = (im_patho_data - np.min(im_patho_data)) / (np.max(im_patho_data) - np.min(im_patho_data))

    """
    Main logic - copy lesion from pathological image to healthy image
    """
    # Initialize Image instances for the new target and lesion
    new_target = zeros_like(im_healthy)
    new_lesion = zeros_like(im_healthy)
    # Initialize numpy arrays with the same shape as the healthy image
    new_target_data = np.copy(im_healthy_data)
    new_lesion_data = np.zeros_like(im_healthy_data)

    # Create a copy of the healthy SC mask. The mask will have the proper output name and will be saved under masksTr
    # folder. The mask is useful for lesion QC (using sct_qc) or nnU-Net region-based training
    # (https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/region_based_training.md)
    new_sc = im_healthy_sc.copy()

    # Check if label_patho has non-zero pixels (i.e., there is no lesion). If so, skip this subject because there is
    # nothing to copy from the pathological image
    if np.count_nonzero(im_patho_lesion_data) == 0:
        print(f"Warning: {path_label_patho} has no non-zero pixels (i.e. no lesion) --> skipping subject\n")
        return

    # Create 3D bounding box around non-zero pixels (i.e., around the lesion)
    coords = np.argwhere(im_patho_lesion_data > 0)

    # Get centerline from healthy SC seg. The centerline is used to project the lesion from the pathological image
    healthy_centerline = get_centerline(im_healthy_sc_data)
    # Make sure that the z-axis is at the max of the SC mask so that it is not mapped on the brainstem
    healthy_centerline_cropped = healthy_centerline[round(len(healthy_centerline)*0.1):
                                                    round(len(healthy_centerline)*0.9)]

    # Select random coordinate on the centerline
    # index is used to have different seed for every subject to have different lesion positions across different
    # subjects
    rng = np.random.default_rng(args.seed + index)

    while True:
        # New position for the lesion
        new_position = healthy_centerline_cropped[rng.integers(0, len(healthy_centerline_cropped) - 1)]
        # x, y, z = healthy_centerline_cropped[rng.integers(0, len(healthy_centerline_cropped) - 1)]
        print(f"Trying to insert lesion at {new_position}")

        # Insert lesion from the bounding box to the new_target
        new_target_data, new_lesion_data = insert_lesion(new_target_data, new_lesion_data, im_patho_data,
                                                         im_patho_lesion_data, im_healthy_sc_data, coords, new_position,
                                                         intensity_ratio)

        # Check if lesion was inserted, i.e., new_lesion contains non-zero pixels
        if np.count_nonzero(new_lesion) > 0:
            print(f"Lesion inserted at {new_position}")
            break

    # Insert newly created target and lesion into Image instances
    new_target.data = new_target_data
    new_lesion.data = new_lesion_data

    if args.resample:
        # Resample new_target and new_lesion to the spacing of pathology subject
        print(f'Resampling new_target and new_lesion to the spacing of pathology subject ({path_image_patho}).')

        print(f'Before resampling: {new_target.dim[4:7]}')

        # Fetch voxel size of pathology subject (will be used for resampling)
        # Note: get_zooms() is nibabel function that returns voxel size in mm (same as SCT's im_patho.dim[4:7])
        im_patho_voxel_size = im_patho.header.get_zooms()

        # Resample
        new_target = resample_nib(new_target, new_size=im_patho_voxel_size, new_size_type='mm', interpolation='linear')
        new_lesion = resample_nib(new_lesion, new_size=im_patho_voxel_size, new_size_type='mm', interpolation='linear')
        new_sc = resample_nib(new_sc, new_size=im_patho_voxel_size, new_size_type='mm', interpolation='linear')

        print(f'After resampling: {new_target.dim[4:7]}')

    # Convert i to string and add 3 leading zeros
    s = str(index)
    s = s.zfill(3)

    """
    Save new_target and new_lesion
    """
    subject_name_out = sub_healthy.split('_')[0] + '_' + \
                       sub_patho.split('_')[0] + '_' + \
                       sub_patho.split('_')[1] + '_' + s

    new_target_path = os.path.join(args.dir_healthy, subject_name_out + '_0000.nii.gz')
    new_lesion_path = os.path.join(args.dir_save, subject_name_out + '.nii.gz')
    new_sc_path = os.path.join(args.dir_masks_healthy, subject_name_out + '.nii.gz')

    new_target.save(new_target_path)
    print(f'Saving {new_target_path}; {new_target.orientation, new_target.dim[4:7]}')
    new_lesion.save(new_lesion_path)
    print(f'Saving {new_lesion_path}; {new_lesion.orientation, new_lesion.dim[4:7]}')
    new_sc.save(new_sc_path)
    print(f'Saving {new_sc_path}; {new_sc.orientation, new_sc.dim[4:7]}')
    print('')

    # Generate QC
    if args.qc:
        # Binarize new_lesion (sct_qc supports only binary masks)
        new_lesion_bin_path = new_lesion_path.replace('.nii.gz', '_bin.nii.gz')
        os.system(f'sct_maths -i {new_lesion_path} -bin 0 -o {new_lesion_bin_path}')
        # Example: sct_qc -i t2.nii.gz -s t2_seg.nii.gz -d t2_lesion.nii.gz -p sct_deepseg_lesion -plane axial
        os.system(f'sct_qc -i {new_target_path} -s {new_sc_path} -d {new_lesion_bin_path} -p sct_deepseg_lesion '
                  f'-plane sagittal -qc {args.dir_save.replace("labelsTr", "qc")} -qc-subject {subject_name_out}')
        # Remove binarized lesion
        os.remove(new_lesion_bin_path)


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Expand user (i.e. ~) in paths
    args.dir_healthy = os.path.expanduser(args.dir_healthy)
    args.dir_masks_healthy = os.path.expanduser(args.dir_masks_healthy)
    args.dir_pathology = os.path.expanduser(args.dir_pathology)
    args.dir_lesions = os.path.expanduser(args.dir_lesions)
    args.dir_masks_pathology = os.path.expanduser(args.dir_masks_pathology)
    args.dir_save = os.path.expanduser(args.dir_save)

    # if not os.path.exists(args.mask_check_path):
    #     os.makedirs(args.mask_check_path, exist_ok=True)

    # get all pathology cases
    # TODO - maybe could be changed to args.dir_pathology
    cases_patho = os.listdir(args.dir_lesions)
    # remove '.DS_Store' from Cases list
    if '.DS_Store' in cases_patho:
        cases_patho.remove('.DS_Store')
    simple_cases_patho = [case.split('.')[0] for i, case in enumerate(cases_patho) if 'Mix' not in case]
    cases_patho = simple_cases_patho

    # get all healthy cases
    # TODO - maybe could be changed to args.dir_healthy
    cases_healthy = os.listdir(args.dir_masks_healthy)
    # remove '.DS_Store' from Cases list
    if '.DS_Store' in cases_healthy:
        cases_healthy.remove('.DS_Store')
    simple_cases_healthy = [case.split('.')[0] for i, case in enumerate(cases_healthy) if 'Mix' not in case]
    cases_healthy = simple_cases_healthy

    # Check if number of samples to generate is not larger than the number of available subjects
    # Because we want to use each healthy subject only once
    if args.num > len(cases_healthy):
        sys.exit(f"Number of samples to generate ({args.num}) is larger than the number of available "
                 f"subjects ({len(cases_patho)})")

    """
    Mix pathology and healthy subjects
    """
    print("Random seed: ", args.seed)
    rng = np.random.default_rng(args.seed)
    # Get random indices for pathology and healthy subjects
    patho_random_list = rng.choice(len(cases_patho), args.num)
    healthy_random_list = rng.choice(len(cases_healthy), args.num, replace=False)
    # Combine both lists
    rand_index = np.vstack((patho_random_list, healthy_random_list))
    # Keep only unique combinations (to avoid mixing the same subjects)
    rand_index = np.unique(rand_index, axis=1)

    """
    Start generating new samples
    """
    for i in tqdm(range(len(rand_index[0])), desc="mixing:"):

        # wait 0.1 seconds to avoid print overlapping with tqdm progress bar
        time.sleep(0.1)

        rand_index_patho = rand_index[0][i]
        rand_index_healthy = rand_index[1][i]

        sub_patho = cases_patho[rand_index_patho]
        sub_healthy = cases_healthy[rand_index_healthy]

        print(f"\nHealthy subject: {sub_healthy}, Patho subject: {sub_patho}")

        generate_new_sample(sub_healthy=sub_healthy, sub_patho=sub_patho, args=args, index=i)

        # wait 0.1 seconds to avoid print overlapping with tqdm progress bar
        time.sleep(0.1)


if __name__ == '__main__':
    main()
