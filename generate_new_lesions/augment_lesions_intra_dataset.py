"""
Use lesions from pathology subjects in dataset A and insert them into healthy controls within dataset A.

Activate SCT conda environment:
    source ${SCT_DIR}/python/etc/profile.d/conda.sh
    conda activate venv_sct

Run:
    python augment_lesions_intra_dataset.py

NOTE: BIDS structure is expected for the input data.
"""
import os
import sys
import time
import yaml
import argparse
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, generate_binary_structure
from sklearn.model_selection import train_test_split

from spinalcordtoolbox.image import Image, zeros_like
from spinalcordtoolbox.resampling import resample_nib

from utils import get_centerline, get_lesion_volume, fetch_subject_and_session, \
    generate_histogram, extract_lesions

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path-data", type=str, required=True,
                        help="Path to BIDS dataset containing both healthy controls and patients with lesions "
                             "(e.g., basel-mp2rage)")
    parser.add_argument("-path-out", type=str, required=True,
                        help="Path where the augmented dataset will be saved.")
    parser.add_argument("-num", default=100, type=int, help="Total number of newly generated subjects. Default: 100")
    parser.add_argument("-seed", default=99, type=int, help="Random seed used for subject mixing. Default: 99")
    parser.add_argument("-resample", default=False, action='store_true',
                        help="Resample the augmented images to the resolution of pathological dataset. Default: False")
    parser.add_argument("-use-n-healthy", default=1, type=int,
                        help="Number of healthy controls to use for augmentation Default: num_patho // 2")
    parser.add_argument('--fraction_patho_data', default=1.0, type=float,
                    help='Fraction of pathology data to be used for training. '
                    'Use 0.5 for half, 0.25 for 1/4th, etc. ') 
    parser.add_argument('-split', nargs='+', required=False, type=float, default=[0.6, 0.2, 0.2],
                        help='Ratios of training, validation and test splits lying between 0-1. Example: --split 0.6 0.2 0.2')
    parser.add_argument("-qc", default=False, action='store_true', 
                        help="Perform QC using sct_qc. QC will be saved under -path-out. Default: False")
    # parser.add_argument("-histogram", default=False, action='store_true', help="Create histograms. Default: False")
    # parser.add_argument("-min-lesion-vol", "--min-lesion-volume", default=200, type=float,
    #                     help="Minimum lesion volume in mm^3. Default: 200")

    return parser


def insert_lesion(im_augmented, im_augmented_lesion, im_patho_data, im_patho_sc_dil_data, patho_lesion_data, 
                  im_healthy_sc_data, coords, new_position, lesion_sc_ratio_patho):
    """"
    Insert lesion from the bounding box to the im_augmented
    """
    # Get bounding box coordinates
    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0) + 1  # slices are exclusive at the top

    # Get coordinates where to insert the lesion
    x, y, z = new_position

    # Compute the shift to the center of the bounding box
    shift_x, shift_y, shift_z = (x1 - x0) // 2, (y1 - y0) // 2, (z1 - z0) // 2

    # The new origin is shifted to the center of the bounding box (unlike the bottom left earlier)
    for x_step, x_cor in zip(range(-shift_x, shift_x + 1), range(x0, x1)):
        for y_step, y_cor in zip(range(-shift_y, shift_y + 1 ), range(y0, y1)):
            for z_step, z_cor in zip(range(-shift_z, shift_z + 1), range(z0, z1)):

                # make sure that the new lesion is not projected outside of the SC
                if patho_lesion_data[x_cor, y_cor, z_cor] > 0 and im_healthy_sc_data[x + x_step, y + y_step, z + z_step] > 0:
                    # ensure that patho_lesion does not overlap with an existing lesion in label
                    if im_augmented_lesion[x + x_step, y + y_step, z + z_step] == 0:
                        # Simply copy the lesion voxels
                        im_augmented[x + x_step, y + y_step, z + z_step] = im_patho_data[x_cor, y_cor, z_cor]
                                                                
                        # Lesion mask
                        im_augmented_lesion[x + x_step, y + y_step, z + z_step] = patho_lesion_data[x_cor, y_cor, z_cor]
                    else: 
                        continue

    # dilate the augmented lesion mask same as before
    im_augmented_lesion_dilated = im_augmented_lesion.copy()
    im_augmented_lesion_dilated = binary_dilation(im_augmented_lesion_dilated, structure=generate_binary_structure(3, 5), iterations=3)
    # extract only the dilated region of the lesion from the healthy SC
    im_healthy_sc_dil_data = im_healthy_sc_data * im_augmented_lesion_dilated
    # print(f"non zero elements in healthy SC after dilation: {np.count_nonzero(im_healthy_sc_data)}")

    # compute the intensity ratio of the SCs of the patho and healthy image
    intensity_ratio_scs = np.mean(im_augmented[(im_healthy_sc_dil_data > 0) & (im_augmented_lesion == 0)]) / \
                                    np.mean(im_patho_data[(im_patho_sc_dil_data > 0) & (patho_lesion_data == 0)])  # without lesion
    # print(f"Mean Intensity ratio of augmented/patho SC: {intensity_ratio_scs}")

    # modify the augmented lesion voxels with the intensity ratio
    im_augmented[im_augmented_lesion > 0] = im_augmented[im_augmented_lesion > 0] * intensity_ratio_scs

    # compute the lesion/SC intensity ratio in the augmented image
    lesion_sc_ratio_augmented = np.mean(im_augmented[im_augmented_lesion > 0]) / np.mean(im_augmented[(im_healthy_sc_dil_data > 0) & (im_augmented_lesion == 0)])   # without lesion
    print(f"Mean Lesion/SC Intensity Ratio of Augmented Subject (AFTER LESION INSERTION): {lesion_sc_ratio_augmented}")

    # modify the intensity ratio of augmented lesion to be similar to that of the patho lesion
    im_augmented[im_augmented_lesion > 0] = im_augmented[im_augmented_lesion > 0] * lesion_sc_ratio_patho / lesion_sc_ratio_augmented

    # recompute the lesion/SC intensity ratio in the augmented image
    lesion_sc_ratio_augmented = np.mean(im_augmented[im_augmented_lesion > 0]) / np.mean(im_augmented[(im_healthy_sc_dil_data > 0) & (im_augmented_lesion == 0)])
    print(f"Mean Lesion/SC Intensity Ratio of Augmented Subject (AFTER INTENSITY MODIFICATION): {lesion_sc_ratio_augmented}")

    return im_augmented, im_augmented_lesion, im_healthy_sc_dil_data


def generate_new_sample(sub_healthy, sub_patho, args, index):

    # TODO: Create classes for healthy and pathology subjects --> Requires major refactoring

    """
    Load pathological subject image, spinal cord segmentation, and lesion mask
    """
    # Construct paths
    path_image_patho = os.path.join(args.path_data, sub_patho, "anat", f"{sub_patho}_UNIT1.nii.gz")
    path_label_patho = os.path.join(args.path_data, "derivatives", "labels", sub_patho, "anat", f"{sub_patho}_UNIT1_lesion-manualNeuroPoly.nii.gz")
    path_mask_sc_patho = os.path.join(args.path_data, "derivatives", "labels", sub_patho, "anat", f"{sub_patho}_UNIT1_label-SC_seg.nii.gz")

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

    # Check if the patho subject has a lesion, if not, skip this subject
    if not np.any(im_patho_lesion_data):
        print("WARNING: no lesion in image_patho --> skipping subject\n")
        return False

    # # Check if lesion volume is less than X mm^3, if yes, skip this subject
    # im_patho_lesion_vol = get_lesion_volume(im_patho_lesion_data, im_patho.dim[4:7], debug=False)
    # # if im_patho_lesion_vol < args.min_lesion_volume:
    # #     print("WARNING: lesion volume is too small --> skipping subject\n")
    # #     return False


    """
    Load healthy subject image and spinal cord segmentation
    """
    # Construct paths
    path_image_healthy = os.path.join(args.path_data, sub_healthy, "anat", f"{sub_healthy}_UNIT1.nii.gz")
    path_mask_sc_healthy = os.path.join(args.path_data, "derivatives", "labels", sub_healthy, "anat", f"{sub_healthy}_UNIT1_label-SC_seg.nii.gz")

    # Load image_healthy and mask_sc
    im_healthy, im_healthy_sc = Image(path_image_healthy), Image(path_mask_sc_healthy)

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
        # print(f'Resampling healthy subject image and SC mask to the spacing of pathology subject ({path_image_patho}).')

        # print(f'Before resampling - Image Shape: {im_healthy.dim[0:3]}, Image Resolution: {im_healthy.dim[4:7]}')
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

    # Get numpy arrays
    im_healthy_data = im_healthy.data
    im_healthy_sc_data = im_healthy_sc.data

    # Check if image and spinal cord mask have the same shape, if not, skip this subject
    if im_healthy_data.shape != im_healthy_sc_data.shape:
        print("Warning: image_healthy and mask_sc have different shapes --> skipping subject")
        return False

    # Extract all individual lesions from the pathological image
    extracted_patho_lesions = extract_lesions(im_patho_lesion_data)


    """
    Main logic - copy lesion from pathological image to healthy image
    """
    # Initialize Image instances for the new target and lesion
    im_augmented = zeros_like(im_healthy)
    im_augmented_lesion = zeros_like(im_healthy)

    # Create a copy of the healthy SC mask. The mask will have the proper output name and will be saved under masksTr
    # folder. The mask is useful for lesion QC (using sct_qc) or nnU-Net region-based training
    # (https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/region_based_training.md)
    new_sc = im_healthy_sc.copy()

    # Get centerline from healthy SC seg. The centerline is used to project the lesion from the pathological image
    healthy_centerline = get_centerline(im_healthy_sc_data)

    # Initialize numpy arrays with the same shape as the healthy image once
    im_augmented_data = np.copy(im_healthy_data)
    im_augmented_lesion_data = np.zeros_like(im_healthy_data)

    # Select random coordinate on the centerline
    # index is used to have different seed for every subject to have different lesion positions across different subjects
    rng = np.random.default_rng(args.seed + index)

    for i, patho_lesion_data in enumerate(extracted_patho_lesions):
        print(f"Inserting Lesion {i+1} out of {len(extracted_patho_lesions)}")
        """
        Get intensity ratio between healthy and pathological SC and normalize images.
        The ratio is used to multiply the lesion in the healthy image.
        """    
        # Extract SC mask in the neighbourhood of the lesion
        patho_lesion_data_dilated = binary_dilation(patho_lesion_data, structure=generate_binary_structure(3, 5), iterations=3)
        im_patho_sc_dil_data = im_patho_sc_data * patho_lesion_data_dilated
        
        # compute the ratio of intensities between lesion and SC in the pathological image
        lesion_sc_ratio_patho = np.mean(im_patho_data[patho_lesion_data > 0]) / np.mean(im_patho_data[(im_patho_sc_dil_data > 0) & (patho_lesion_data == 0)])    # without lesion
        print(f"Mean lesion/SC Intensity Ratio of Patho Subject {sub_patho}: {lesion_sc_ratio_patho}")
        # Make sure the intensity ratio is always < 1 (i.e. the lesion is always darker (because mp2rage) than the healthy SC)
        if lesion_sc_ratio_patho > 1:
            lesion_sc_ratio_patho = 1 / lesion_sc_ratio_patho
            print(f"Mean lesion/SC Intensity Ratio of Patho Subject {sub_patho} (after inversion): {lesion_sc_ratio_patho}")

        # Create 3D bounding box around non-zero pixels (i.e., around the lesion)
        lesion_coords = np.argwhere(patho_lesion_data > 0)

        # NOTE: This loop is required because the lesion from the original patho image could be cropped if it is going
        # outside of the SC in the healthy image. So, the loop continues until the lesion inserted in the healthy image
        # has similar volume w.r.t the original lesion
        j = 0
        while True:            
            # Initialize numpy arrays with the same shape as the healthy image
            im_augmented_data_new = np.copy(im_augmented_data)
            im_augmented_lesion_data_new = np.copy(im_augmented_lesion_data)

            # get the volume of the augmented lesion first
            lesion_vol_aug_init = get_lesion_volume(im_augmented_lesion_data_new, im_augmented_lesion.dim[4:7], debug=False)

            # New position for the lesion (on the centerline)
            new_position = healthy_centerline[rng.integers(0, len(healthy_centerline) - 1)]
            # # New position for the lesion (randomly selected from the lesion coordinates)
            # new_position = new_coords[rng.integers(0, len(new_coords) - 1)]
            print(f"Trying to insert lesion at {new_position}")

            # Insert lesion from the bounding box to the im_augmented
            im_augmented_data_new, im_augmented_lesion_data_new, _ = insert_lesion(im_augmented_data_new, im_augmented_lesion_data_new, im_patho_data,
                                                            im_patho_sc_dil_data, patho_lesion_data, im_healthy_sc_data,
                                                            lesion_coords, new_position, lesion_sc_ratio_patho)

            # check if im_augmented_lesion_data is empty and if so, try again (with a different position)
            if not im_augmented_lesion_data_new.any():
                print(f"Lesion inserted at {new_position} is empty. Trying again...")
                continue
            
            # NOTE: Keeping largest component for MS lesions suppresses a lot of small lesions. Hence, unlike in SCI, we 
            # don't use it here. 

            # Check if volume of inserted lesion is similar to that of the original one 
            lesion_vol_aug_final = get_lesion_volume(im_augmented_lesion_data_new, im_augmented_lesion.dim[4:7], debug=False)
            print(f"Volume of lesion after augmentation: {lesion_vol_aug_final - lesion_vol_aug_init}")
            lesion_vol_orig = get_lesion_volume(patho_lesion_data, im_patho_lesion.dim[4:7], debug=False)
            print(f"Volume of lesion {i+1} in original image: {lesion_vol_orig}")
            # keep the augmented lesion if it is greater than X % of the original lesion
            if (lesion_vol_aug_final - lesion_vol_aug_init) >= 0.9 * lesion_vol_orig:
                print(f"Lesion inserted at {new_position}")
                break

            if j == 10:
                print(f"WARNING: Tried 10 times to insert lesion but failed. Skipping this lesion...")
                break
            j += 1

        # re-use im_augmented_data for the next lesion
        im_augmented_data = np.copy(im_augmented_data_new)
        im_augmented_lesion_data = np.copy(im_augmented_lesion_data_new)

        
    # Insert newly created target and lesion into Image instances
    im_augmented.data = im_augmented_data
    im_augmented_lesion.data = im_augmented_lesion_data

    # Add a final check to ensure that the im_augmented_lesion is not empty
    if np.sum(im_augmented_lesion.data) == 0:
        print(f"WARNING: (augmented) im_augmented_lesion is empty. Check code again. Gracefully exiting....")
        sys.exit(1)

    """
    Save im_augmented and im_augmented_lesion
    """
    # # Get subject and session IDs from the healthy image
    # subjectID_healthy, sessionID_healthy, _ = fetch_subject_and_session(sub_healthy)
    # # Get subject and session IDs from the patho image
    # subjectID_patho, sessionID_patho, _ = fetch_subject_and_session(sub_patho)

    # if sessionID_patho is None:
    #     subject_name_out = subjectID_healthy + '_' + subjectID_patho + '_augmented'
    # # NOTE: Zurich also has sessions (e.g. sub-zh11_ses-01)
    # else:
    #     subject_name_out = subjectID_healthy + '_' + subjectID_patho + '_' + sessionID_patho + '_augmented'

    subject_name_out = sub_healthy + '_' + sub_patho

    # if args.histogram:
    #     # Generate healthy-patho pair histogram
    #     generate_histogram(im_healthy_data, im_healthy_sc_data, im_healthy_sc_dil_data,
    #                        im_patho_data, im_patho_sc_data, im_patho_sc_dil_data, im_patho_lesion_data,
    #                        im_augmented_data, im_augmented_lesion_data, new_sc.data,
    #                        sub_healthy, sub_patho, subject_name_out,
    #                        output_dir=args.dir_save.replace("labelsTr", "histograms"))

    # if subjectID_patho.startswith('sub-zh'):
    qc_plane = 'sagittal'
    # else:
    #     qc_plane = 'axial'
    
    # save the augmented data in a separate folder with the seed used
    dataset_name = args.path_data.split('/')[-1]
    save_path = os.path.join(args.path_out, f"augmented-{dataset_name}-seed{args.seed}")
    os.makedirs(save_path, exist_ok=True)
    # create new folder for each subject
    sub_save_path = os.path.join(save_path, subject_name_out)
    os.makedirs(sub_save_path, exist_ok=True)
    # save the augmented data in the subject folder
    im_augmented_path = os.path.join(sub_save_path, f"{subject_name_out}_UNIT1_augmented.nii.gz")
    im_augmented_lesion_path = os.path.join(sub_save_path, f"{subject_name_out}_UNIT1_lesion-augmented.nii.gz")
    new_sc_path = os.path.join(sub_save_path, f"{subject_name_out}_UNIT1_label-SC_seg-augmented.nii.gz")

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
                  f'-plane {qc_plane} -qc {os.path.join(save_path, "qc")} -qc-subject {subject_name_out}')
        # Remove binarized lesion
        os.remove(im_augmented_lesion_bin_path)

    return True


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Expand user (i.e. ~) in paths
    args.path_data = os.path.expanduser(args.path_data)

    # define random number generator
    print("Random seed: ", args.seed)
    rng = np.random.default_rng(args.seed)

    # get split ratios
    train_ratio, val_ratio, test_ratio = args.split

    # get all subjects from participants.tsv
    subjects_df = pd.read_csv(os.path.join(args.path_data, 'participants.tsv'), sep='\t')
    subjects = subjects_df['participant_id'].values.tolist()
    print(f"Total number of subjects in the dataset: {len(subjects)}")

    cases_patho = [sub for sub in subjects if sub.startswith('sub-P')]
    print(f"Total number of pathology subjects: {len(cases_patho)}")

    cases_healthy = [sub for sub in subjects if sub.startswith('sub-C')]
    print(f"Total number of healthy subjects: {len(cases_healthy)}")

    # choose a random set of healthy subjects for augmentation
    selected_healthy_subjects = rng.choice(cases_healthy, args.use_n_healthy)

    # Get only the training and test splits initially from the pathology cases only
    train_subjects, test_subjects = train_test_split(cases_patho, test_size=test_ratio, random_state=args.seed)

    # choose only fraction of the train_subjects (which are pathology) for augmentation
    train_subjects = rng.choice(train_subjects, int(len(train_subjects) * args.fraction_patho_data), replace=False).tolist()
    print(f"Using {int(args.fraction_patho_data * 100)}% of pathology subjects for augmentation.")

    # Use the training split to further split into training and validation splits
    train_subjects, val_subjects = train_test_split(train_subjects, test_size=val_ratio / (train_ratio + val_ratio),
                                                    random_state=args.seed, )
    # # add the healthy subjects to the training set
    # train_subjects = train_subjects + cases_healthy     # selected_healthy_subjects

    # create a yml file with the train, val and test subjects
    split_dict = {'train': train_subjects, 'val': val_subjects, 'test': test_subjects}
    split_dict['seed'] = args.seed

    """
    Mix pathology and healthy subjects
    """
    # Get random indices for pathology and healthy subjects
    patho_random_list = rng.choice(len(train_subjects), len(cases_healthy)) 
    healthy_random_list = rng.choice(len(cases_healthy), len(cases_healthy))     # rng.choice(len(cases_healthy), args.num, replace=False)

    # Combine both lists
    rand_index = np.vstack((patho_random_list, healthy_random_list))
    # # Keep only unique combinations (to avoid mixing the same subjects)
    # rand_index = np.unique(rand_index, axis=1)
    # # np.unique sorts the array, so we need to shuffle it again
    # rng.shuffle(rand_index.T)

    num_of_samples_generated = 0

    """
    Start generating new samples
    """
    for i in range(rand_index.shape[1]):

        # wait 0.1 seconds to avoid print overlapping
        time.sleep(0.1)

        rand_index_patho = rand_index[0][i]
        rand_index_healthy = rand_index[1][i]

        sub_patho = cases_patho[rand_index_patho]
        # sub_patho = patho_random_list[rand_index_patho]
        sub_healthy = cases_healthy[rand_index_healthy]
        
        print(f"\nHealthy subject: {sub_healthy}, Patho subject: {sub_patho}")

        # If augmentation is done successfully (True is returned), break the loop and continue to the next sample
        # If augmentation is not done successfully (False is returned), continue the while loop and try again
        if generate_new_sample(sub_healthy=sub_healthy, sub_patho=sub_patho, args=args, index=i):
            num_of_samples_generated += 1
            print('-' * 50)
            print(f"Number of samples generated: {num_of_samples_generated}/{args.num}")
            print('-' * 50)
            
            # add the augmented subject to the yml file
            split_dict['train'].append(f'{sub_healthy}_{sub_patho}')
            

        # If we have generated the required number of samples, break the for loop
        if num_of_samples_generated == args.num:
            break

        # wait 0.1 seconds to avoid print overlapping
        time.sleep(0.1)

    print("\nFinished generating new samples!")

    split_file = os.path.join(args.path_out, 'data_split.yml')
    with open(split_file, 'w') as outfile:
        yaml.dump(split_dict, outfile, default_flow_style=False)


if __name__ == '__main__':
    main()
