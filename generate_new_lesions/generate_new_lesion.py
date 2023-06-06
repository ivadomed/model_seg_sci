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
import os
import sys
import time
import argparse
import numpy as np

from spinalcordtoolbox.image import Image, zeros_like
from spinalcordtoolbox.resampling import resample_nib

from utils import coefficient_of_variation, get_centerline, get_lesion_volume, keep_largest_component

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
    parser.add_argument("-min-lesion-vol", "--min-lesion-volume", default=200, type=float,
                        help="Minimum lesion volume in mm^3. Default: 200")
    # parser.add_argument("--mask_save_path", "-mask-pth", default="mask", type=str,
    #                     help="Path to save carved masks")

    return parser


def insert_lesion(im_augmented, im_augmented_lesion, im_patho_data, im_patho_lesion_data, im_healthy_sc_data, coords,
                  new_position, intensity_ratio):
    """"
    Insert lesion from the bounding box to the im_augmented
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
                if x + x_step >= im_augmented.shape[0] or y + y_step >= im_augmented.shape[1] or z + z_step >= im_augmented.shape[2]:
                    continue
                # Insert only voxels corresponding to the lesion mask
                # Also make sure that the new lesion is not projected outside of the SC
                if im_patho_lesion_data[x_cor, y_cor, z_cor] > 0 and im_healthy_sc_data[x + x_step, y + y_step, z + z_step] > 0:
                    # Lesion inserted into the target image
                    im_augmented[x + x_step, y + y_step, z + z_step] = im_patho_data[x_cor, y_cor, z_cor] * intensity_ratio
                    # Lesion mask
                    im_augmented_lesion[x + x_step, y + y_step, z + z_step] = im_patho_lesion_data[x_cor, y_cor, z_cor]

    return im_augmented, im_augmented_lesion



def generate_new_sample(sub_healthy, sub_patho, args, index):

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
    print(f"Pathological subject {path_image_patho}: {im_patho_orientation_native}, {im_patho.dim[0:3]}, {im_patho.dim[4:7]}")
    # TODO: consider reorienting back to native orientation

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
        print("WARNING: image_patho and label_patho have different shapes --> skipping subject\n")
        return False

    # Check if lesion volume is less than X mm^3, if yes, skip this subject
    im_patho_lesion_vol = get_lesion_volume(im_patho_lesion_data, im_patho.dim[4:7], debug=False)
    if im_patho_lesion_vol < args.min_lesion_volume:
        print("WARNING: lesion volume is too small --> skipping subject\n")
        return False


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
    print(f"Healthy subject {path_image_healthy}: {im_healthy_orientation_native}, {im_healthy.dim[0:3]}, {im_healthy.dim[4:7]}")

    # Reorient to RPI
    im_healthy.change_orientation("RPI")
    im_healthy_sc.change_orientation("RPI")
    print("Reoriented to RPI")


    """
    Resample healthy subject image and spinal cord mask to the spacing of pathology subject
    """
    if args.resample:
        # Resample healthy subject to the spacing of pathology subject
        print(f'Resampling healthy subject image and SC mask to the spacing of pathology subject ({path_image_patho}).')

        print(f'Before resampling - Image Shape: {im_healthy.dim[0:3]}, Image Resolution: {im_healthy.dim[4:7]}')
        # new_lesion_vol = get_lesion_volume(new_lesion.data, new_lesion.dim[4:7], debug=False)
        # print(f'Lesion volume before resampling: {new_lesion_vol} mm3')

        # Fetch voxel size of pathology subject (will be used for resampling)
        # Note: get_zooms() is nibabel function that returns voxel size in mm (same as SCT's im_patho.dim[4:7])
        im_patho_voxel_size = im_patho.header.get_zooms()

        # Resample
        # Note: we cannot use 'image_dest=' option because we do not want to introduce padding or cropping
        im_healthy = resample_nib(im_healthy, new_size=im_patho_voxel_size, new_size_type='mm', interpolation='linear')
        # new_lesion = resample_nib(new_lesion, new_size=im_patho_voxel_size, new_size_type='mm', interpolation='linear')
        im_healthy_sc = resample_nib(im_healthy_sc, new_size=im_patho_voxel_size, new_size_type='mm', interpolation='linear')
        # new_sc = resample_nib(new_sc, new_size=im_patho_voxel_size, new_size_type='mm', interpolation='linear')

        print(f'After resampling - Image Shape: {im_healthy.dim[0:3]}, Image Resolution: {im_healthy.dim[4:7]}')
        # print(f'After resampling - SC Shape: {im_healthy_sc.dim[0:3]}, SC Resolution: {im_healthy_sc.dim[4:7]}')
        # new_lesion_vol = get_lesion_volume(new_lesion.data, new_lesion.dim[4:7], debug=False)
        # print(f'Lesion volume before resampling: {new_lesion_vol} mm3')
        # TODO: lesion volume is not the same after resampling. And can be even zero!

    # Get numpy arrays
    im_healthy_data = im_healthy.data
    im_healthy_sc_data = im_healthy_sc.data

    # Check if image and spinal cord mask have the same shape, if not, skip this subject
    if im_healthy_data.shape != im_healthy_sc_data.shape:
        print("Warning: image_healthy and mask_sc have different shapes --> skipping subject")
        return False


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
    im_augmented = zeros_like(im_healthy)
    im_augmented_lesion = zeros_like(im_healthy)
    # Initialize numpy arrays with the same shape as the healthy image
    im_augmented_data = np.copy(im_healthy_data)
    im_augmented_lesion_data = np.zeros_like(im_healthy_data)

    # Create a copy of the healthy SC mask. The mask will have the proper output name and will be saved under masksTr
    # folder. The mask is useful for lesion QC (using sct_qc) or nnU-Net region-based training
    # (https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/region_based_training.md)
    new_sc = im_healthy_sc.copy()

    # Create 3D bounding box around non-zero pixels (i.e., around the lesion)
    lesion_coords = np.argwhere(im_patho_lesion_data > 0)

    # Get centerline from healthy SC seg. The centerline is used to project the lesion from the pathological image
    healthy_centerline = get_centerline(im_healthy_sc_data)
    # Make sure that the z-axis is at the max of the SC mask so that it is not mapped on the brainstem
    healthy_centerline_cropped = healthy_centerline[round(len(healthy_centerline)*0.2):
                                                    round(len(healthy_centerline)*0.9)]

    # Select random coordinate on the centerline
    # index is used to have different seed for every subject to have different lesion positions across different
    # subjects
    rng = np.random.default_rng(args.seed + index)

    # NOTE: This loop is required because the lesion from the original patho image could be cropped if it is going
    # outside of the SC in the healthy image. So, the loop continues until the lesion inserted in the healthy image
    # is greater than args.min_lesion_volume
    while True:
        # New position for the lesion
        new_position = healthy_centerline_cropped[rng.integers(0, len(healthy_centerline_cropped) - 1)]
        # x, y, z = healthy_centerline_cropped[rng.integers(0, len(healthy_centerline_cropped) - 1)]
        print(f"Trying to insert lesion at {new_position}")

        # Insert lesion from the bounding box to the im_augmented
        # TODO: im_augmented and im_patho_lesion have different dimensions and resolution in this step!!!
        #  The reason is because im_augmented was created from im_healthy
        im_augmented_data, im_augmented_lesion_data = insert_lesion(im_augmented_data, im_augmented_lesion_data, im_patho_data,
                                                         im_patho_lesion_data, im_healthy_sc_data, lesion_coords, 
                                                         new_position, intensity_ratio)

        # Inserted lesion can be divided into several parts (due to the crop by the healthy SC mask and due to SC curvature).
        # In such case, keep only the largest part.
        # NOTE: im_augmented_lesion_data could still be empty if the coordinates of lesion bbox are overflowing out of healthy SC, 
        # essentially never reaching the second if statement in insert_lesion() function. As a result, we get 
        # "ValueError: attempt to get argmax of an empty sequence" error in keep_largest_component() function.
        # So, check if im_augmented_lesion_data is empty and if so, try again (with a different position)
        if not im_augmented_lesion_data.any():
            print(f"Lesion inserted at {new_position} is empty. Trying again...")
            continue
        
        im_augmented_lesion_data = keep_largest_component(im_augmented_lesion_data)

        # Insert back intensity values from the original healthy image everywhere where the lesion is zero. In other
        # words, keep only the largest part of the lesion and replace the rest with the original healthy image.
        im_augmented_data[im_augmented_lesion_data == 0] = im_healthy_data[im_augmented_lesion_data == 0]

        # Check if inserted lesion is larger then min_lesion_volume
        # NOTE: we are doing this check because the lesion can smaller due to crop by the spinal cord mask
        lesion_vol = get_lesion_volume(im_augmented_lesion_data, im_augmented_lesion.dim[4:7], debug=False)
        if lesion_vol > args.min_lesion_volume:
            print(f"Lesion inserted at {new_position}")
            break
            # TODO: some subjects have no lesion. explore this!

    # Insert newly created target and lesion into Image instances
    im_augmented.data = im_augmented_data
    im_augmented_lesion.data = im_augmented_lesion_data

    # Add a final check to ensure that the im_augmented_lesion is not empty
    if np.sum(im_augmented_lesion.data) == 0:
        print(f"WARNING: (augmented) im_augmented_lesion is empty. Check code again. Gracefully exiting....")
        sys.exit(1)

    # Convert i to string and add 3 leading zeros
    s = str(index).zfill(3)

    """
    Save im_augmented and im_augmented_lesion
    """
    subject_name_out = sub_healthy.split('_')[0] + '_' + \
                       sub_patho.split('_')[0] + '_' + \
                       sub_patho.split('_')[1] + '_' + s

    im_augmented_path = os.path.join(args.dir_healthy, subject_name_out + '_0000.nii.gz')
    im_augmented_lesion_path = os.path.join(args.dir_save, subject_name_out + '.nii.gz')
    new_sc_path = os.path.join(args.dir_masks_healthy, subject_name_out + '.nii.gz')

    im_augmented.save(im_augmented_path)
    print(f'Saving {im_augmented_path}; {im_augmented.orientation, im_augmented.dim[4:7]}')
    im_augmented_lesion.save(im_augmented_lesion_path)
    print(f'Saving {im_augmented_lesion_path}; {im_augmented_lesion.orientation, im_augmented_lesion.dim[4:7]}')
    new_sc.save(new_sc_path)
    print(f'Saving {new_sc_path}; {new_sc.orientation, new_sc.dim[4:7]}')
    print('')

    # Generate QC
    if args.qc:
        # Binarize im_augmented_lesion (sct_qc supports only binary masks)
        im_augmented_lesion_bin_path = im_augmented_lesion_path.replace('.nii.gz', '_bin.nii.gz')
        os.system(f'sct_maths -i {im_augmented_lesion_path} -bin 0 -o {im_augmented_lesion_bin_path}')
        # Example: sct_qc -i t2.nii.gz -s t2_seg.nii.gz -d t2_lesion.nii.gz -p sct_deepseg_lesion -plane axial
        os.system(f'sct_qc -i {im_augmented_path} -s {new_sc_path} -d {im_augmented_lesion_bin_path} -p sct_deepseg_lesion '
                  f'-plane sagittal -qc {args.dir_save.replace("labelsTr", "qc")} -qc-subject {subject_name_out}')
        # Remove binarized lesion
        os.remove(im_augmented_lesion_bin_path)

    return True


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
    patho_random_list = rng.choice(len(cases_patho), args.num*2) # *2 because we need same number of patho and healthy
    healthy_random_list = rng.choice(len(cases_healthy), args.num, replace=False)

    # Duplicate healthy list (we need more subjects because some pair might be skipped, for example due to no lesion)
    healthy_random_list = np.tile(healthy_random_list, 2)

    # Combine both lists
    rand_index = np.vstack((patho_random_list, healthy_random_list))
    # Keep only unique combinations (to avoid mixing the same subjects)
    rand_index = np.unique(rand_index, axis=1)
    # np.unique sorts the array, so we need to shuffle it again
    rng.shuffle(rand_index.T)

    num_of_samples_generated = 0

    """
    Start generating new samples
    """
    for i in range(len(rand_index[0])):

        # wait 0.1 seconds to avoid print overlapping
        time.sleep(0.1)

        rand_index_patho = rand_index[0][i]
        rand_index_healthy = rand_index[1][i]

        sub_patho = cases_patho[rand_index_patho]
        sub_healthy = cases_healthy[rand_index_healthy]

        # example subjects where augumentation was empty even within the while loop
        # sub_patho = 'sub-zh21_ses-01_017'
        # sub_healthy = 'sub-strasbourg03_192'

        print(f"\nHealthy subject: {sub_healthy}, Patho subject: {sub_patho}")

        # If augmentation is done successfully (True is returned), break the while loop and continue to the next
        # sample
        # If augmentation is not done successfully (False is returned), continue the while loop and try again
        if generate_new_sample(sub_healthy=sub_healthy, sub_patho=sub_patho, args=args, index=i):
            num_of_samples_generated += 1
            print('-' * 50)
            print(f"Number of samples generated: {num_of_samples_generated}/{args.num}")
            print('-' * 50)

        # If we have generated the required number of samples, break the for loop
        if num_of_samples_generated == args.num:
            break

        # wait 0.1 seconds to avoid print overlapping
        time.sleep(0.1)

    print("\nFinished generating new samples!")

if __name__ == '__main__':
    main()
