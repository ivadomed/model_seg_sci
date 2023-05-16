"""
Take lesion from subject_b and insert it into subject_a

Run:
    python generate_new_lesion.py

nnUNet data structure is required.
TODO: switch to BIDS?
"""

import time
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from scipy import ndimage
from tqdm import tqdm
import argparse
import os

# TODO: Check out Diffusion models for synthesizing new images + lesions 

# TODO: Consider moving the script to SCT --> use RPI reorientation function, potentially use angle correction to
#  rotate lesions along the SC, and eventually also SCT QC function to check if the generated lesions


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
    parser.add_argument("-resample", default=False, type=bool, help="Resample the augmented images to the resolution "
                                                                    "of pathological dataset. Default: False")
    # parser.add_argument("--mask_save_path", "-mask-pth", default="mask", type=str,
    #                     help="Path to save carved masks")

    return parser


def get_head(img_path):
    temp = sitk.ReadImage(img_path)
    spacing = temp.GetSpacing()
    direction = temp.GetDirection()
    origin = temp.GetOrigin()

    return spacing, direction, origin


def copy_head_and_right_xyz(data, spacing, direction, origin):
    TrainData_new = data.astype('float32')
    TrainData_new = TrainData_new.transpose(2, 1, 0)
    TrainData_new = sitk.GetImageFromArray(TrainData_new)
    TrainData_new.SetSpacing(spacing)
    TrainData_new.SetOrigin(origin)
    TrainData_new.SetDirection(direction)

    return TrainData_new


def resample_volume(volume, new_spacing, interpolator=sitk.sitkLinear):
    """
    Resample volume to new spacing. Taken from:
    https://discourse.itk.org/t/resample-volume-to-specific-voxel-spacing-simpleitk/3531/2
    :param volume: volume to be resampled
    :param new_spacing: new spacing
    :param interpolator:
    :return:
    """
    # volume = sitk.ReadImage(volume_path, sitk.sitkFloat32) # read and cast to float32
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                         volume.GetOrigin(), new_spacing, volume.GetDirection(), 0,
                         volume.GetPixelID())


def coefficient_of_variation(masked_image):
    return np.std(masked_image, ddof=1) / np.mean(masked_image) * 100


def generate_new_sample(sub_healthy, sub_patho, args, index):

    # Construct paths
    path_image_healthy = os.path.join(args.dir_healthy, sub_healthy + '_0000.nii.gz')
    path_mask_sc_healthy = os.path.join(args.dir_masks_healthy, sub_healthy + '.nii.gz')

    path_image_patho = os.path.join(args.dir_pathology, sub_patho + '_0000.nii.gz')
    path_label_patho = os.path.join(args.dir_lesions, sub_patho + '.nii.gz')
    path_mask_sc_patho = os.path.join(args.dir_masks_pathology, sub_patho + '.nii.gz')

    # get the header of the healthy image
    spacing_healthy, direction_healthy, origin_healthy = get_head(path_image_healthy)
    # get the header of the patho image
    spacing_patho, direction_patho, origin_patho = get_head(path_image_patho)

    # Load image_healthy and mask_sc
    image_healthy = nib.load(path_image_healthy).get_fdata()
    mask_sc = nib.load(path_mask_sc_healthy).get_fdata()

    # Check if image_healthy and mask_sc have the same shape, if not, skip this subject
    if image_healthy.shape != mask_sc.shape:
        print("image_healthy and mask_sc have different shapes")
        return


    # for each slice in the mask_sc, get the center coordinate of the y-axis
    num_z_slices = mask_sc.shape[2]
    centerline = list()
    for z in range(num_z_slices):
        x, y = ndimage.center_of_mass(mask_sc[:, :, z])
        # check if not nan
        if not np.isnan(x) and not np.isnan(y):
            centerline.append((round(x), round(y), z))


    # Load image_patho, label_patho, and mask_sc_patho
    image_patho = nib.load(path_image_patho).get_fdata()
    label_patho = nib.load(path_label_patho).get_fdata()
    mask_sc_patho = nib.load(path_mask_sc_patho).get_fdata()

    # Check if image_healthy and mask_sc have the same shape, if not, skip this subject
    if image_healthy.shape != mask_sc.shape:
        print("image_healthy and mask_sc have different shapes")
        return
    # Check if image_patho and label_patho have the same shape, if not, skip this subject
    if image_patho.shape != mask_sc_patho.shape:
        print("image_patho and label_patho have different shapes")
        return

    # Get intensity ratio healthy/patho SC. This ratio is used to multiply the lesion in the healthy image
    intensity_ratio = coefficient_of_variation(image_healthy[mask_sc > 0]) / \
                      coefficient_of_variation(image_patho[mask_sc_patho > 0])
    # Make sure the intensity ratio is always > 1 (i.e. the lesion is always brighter than the healthy SC)
    if intensity_ratio < 1:
        intensity_ratio = 1 / intensity_ratio

    # normalize images to range 0 and 1
    image_healthy = (image_healthy - np.min(image_healthy)) / (np.max(image_healthy) - np.min(image_healthy))
    image_patho = (image_patho - np.min(image_patho)) / (np.max(image_patho) - np.min(image_patho))

    # Initialize new_target and new_label with the same shape as target_a
    new_target = np.copy(image_healthy)
    new_label = np.zeros_like(mask_sc)  # because we're creating a new label

    # Check if label_patho has non-zero pixels (ie. there is no lesion in the image)
    if np.count_nonzero(label_patho) == 0:
        print(f"label_patho of subject {sub_patho} has no non-zero pixels (i.e. no lesion)")
        return
    # TODO: create an empty lession mask in such case?

    # Create 3D bounding box around non-zero pixels in label_patho
    coords = np.argwhere(label_patho > 0)
    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0) + 1  # slices are exclusive at the top
    
    # get the set of all coordinates where SC mask is 1
    new_coords = np.argwhere(mask_sc > 0)

    # make sure that the z-axis is at the max of the SC mask so that it is not mapped on the brainstem
    new_coords = new_coords[new_coords[:, 2] < int(new_coords[:, 2].max() * 0.9)]

    # Select random coordinate in new_target where SC mask is 1
    rng = np.random.default_rng(args.seed)
    #x, y, z = new_coords[rng.integers(0, len(new_coords) - 1)]
    #print("x, y, z: ", x, y, z)

    # Get random slice (z)
    _, _, z = new_coords[rng.integers(0, len(new_coords) - 1)]
    # Get the center of mass of the spinal cord for the selected random slice
    x, y = ndimage.center_of_mass(mask_sc[:, :, z])
    # Note: we have to round the coordinates because they are floats
    x, y = round(x), round(y)

    x = x - int((x1 - x0) / 2)
    y = y - int((y1 - y0) / 2)
    z = z - int((z1 - z0) / 2)

    # Insert lesion from the bounding box to the new_target
    for x_step, x_cor in enumerate(range(x0, x1)):
        for y_step, y_cor in enumerate(range(y0, y1)):
            for z_step, z_cor in enumerate(range(z0, z1)):
                # # Check that dimensions do not overflow
                # if x + x_step >= new_target.shape[0] or y + y_step >= new_target.shape[1] or z + z_step >= new_target.shape[2]:
                #     continue
                # Insert only voxels corresponding to the lesion mask (label_b)
                # Also make sure that the new lesion is not projected outside of the SC
                if label_patho[x_cor, y_cor, z_cor] > 0 and mask_sc[x + x_step, y + y_step, z + z_step] > 0:
                    new_target[x + x_step, y + y_step, z + z_step] = image_patho[x_cor, y_cor, z_cor] * intensity_ratio
                    new_label[x + x_step, y + y_step, z + z_step] = label_patho[x_cor, y_cor, z_cor]

    # Copy header information from target_a to new_target and new_label
    new_target = copy_head_and_right_xyz(new_target, spacing_healthy, direction_healthy, origin_healthy)
    new_label = copy_head_and_right_xyz(new_label, spacing_healthy, direction_healthy, origin_healthy)

    if args.resample:
        # Resample new_target and new_label to the spacing of pathology subject
        new_target = resample_volume(new_target, new_spacing=spacing_patho)
        new_label = resample_volume(new_label, new_spacing=spacing_patho)

    # Convert i to string and add 3 leading zeros
    s = str(index)
    s = s.zfill(3)

    # Save new_target and new_label
    subject_name_out = sub_healthy.split('_')[0] + '_' + \
                       sub_patho.split('_')[0] + '_' + \
                       sub_patho.split('_')[1] + '_' + s
    sitk.WriteImage(new_target, os.path.join(args.dir_healthy, subject_name_out + '_0000.nii.gz'))
    print('Saving new sample: ', os.path.join(args.dir_healthy, subject_name_out + '_0000.nii.gz'))
    sitk.WriteImage(new_label, os.path.join(args.dir_save, subject_name_out + '.nii.gz'))
    print('Saving new sample: ', os.path.join(args.dir_save, subject_name_out + '.nii.gz'))
    print('')


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

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

    """
    Prepare data split, note that validation sets do not participate in 
    CarveMix, and remember to split training sets and validation sets 
    independently in nnunet.training.network_training.nnUNetTrainerV2.do_split 
    when using nnUNet framework
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

        print("\nPatho subject: ", cases_patho[rand_index_patho], '\t', "Healthy subject: ", cases_healthy[rand_index_healthy])

        sub_patho = cases_patho[rand_index_patho]
        sub_healthy = cases_healthy[rand_index_healthy]

        generate_new_sample(sub_healthy=sub_healthy, sub_patho=sub_patho, args=args, index=i)


if __name__ == '__main__':
    main()
