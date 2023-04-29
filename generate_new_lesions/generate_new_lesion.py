"""
Take lesion from subject_b and insert it into subject_a

Run:
    python generate_new_lesion.py

nnUNet data structure is required.
TODO: switch to BIDS?
"""

import time
import numpy as np
import random
import SimpleITK as sitk
import nibabel as nib
from tqdm import tqdm
import argparse
import os

# TODO: Check out Diffusion models for synthesizing new images + lesions 

# TODO: Consider moving the script to SCT --> use RPI reorientation function, potentially use angle correction to
#  rotate lesions along the SC, and eventually also SCT QC function to check if the generated lesions


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-num", default=5, type=int, help="Number of samples you want to crete.")
    
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


def pad(image, pad_size):
    """Pad the input image to the specified spatial size."""
    pad_width = []
    for i, sp_i in enumerate(pad_size):
        width = max(sp_i - image.shape[i], 0)
        pad_width.append((int(width // 2), int(width - (width // 2))))
    
    to_pad = tuple(pad_width)
    image_padd = np.pad(image, to_pad, mode='constant',)
    
    return image_padd


def pad_or_crop(img_healthy, mask_sc, img_patho, label_patho):

    # size to match after padding
    if img_healthy.shape > img_patho.shape:
        spatial_size = img_healthy.shape
        # print("padding image_b to match the dimensions of image_a")

        # the pad function automatically pads the image to the largest dimensions 
        # between the two images. This might result in shape mismatch hence the next step
        # is to crop the padded image to match the size of the other image
        img_patho = pad(img_patho, spatial_size)
        label_patho = pad(label_patho, spatial_size)
        # print("paddded image_b: ", target_b.shape)

        # TODO: check if this is necessary
        if img_patho.shape != img_healthy.shape:
            # crop the padded image until it matches the size of the other image
            if img_patho.shape[0] > img_healthy.shape[0]:
                img_patho = img_patho[0:img_healthy.shape[0], :, :]
                label_patho = label_patho[0:mask_sc.shape[0], :, :]
            
            if img_patho.shape[1] > img_healthy.shape[1]:
                img_patho = img_patho[:, 0:img_healthy.shape[1], :]
                label_patho = label_patho[:, 0:mask_sc.shape[1], :]

            if img_patho.shape[2] > img_healthy.shape[2]:
                img_patho = img_patho[:, :,  0:img_healthy.shape[2]]
                label_patho = label_patho[:, :, 0:mask_sc.shape[2]]
            
            # print("padded and cropped image_b: ", target_b.shape)

    else:
        spatial_size = img_patho.shape
        # print("padding image_a to match the dimensions of image_b")

        img_healthy = pad(img_healthy, spatial_size)
        mask_sc = pad(mask_sc, spatial_size)
        # print("paddded image_a: ", target_a.shape)

        # TODO: check if this is necessary
        if img_patho.shape != img_healthy.shape:
            # crop the padded image until it matches the size of the other image

            if img_healthy.shape[0] > img_patho.shape[0]:
                img_healthy = img_healthy[0:img_patho.shape[0], :, :]
                mask_sc = mask_sc[0:label_patho.shape[0], :, :]

            if img_healthy.shape[1] > img_patho.shape[1]:
                img_healthy = img_healthy[:, 0:img_patho.shape[1], :]
                mask_sc = mask_sc[:, 0:label_patho.shape[1], :]

            if img_healthy.shape[2] > img_patho.shape[2]:
                img_healthy = img_healthy[:, :,  0:img_patho.shape[2]]
                mask_sc = mask_sc[:, :, 0:label_patho.shape[2]]

            # print("padded and cropped image_a: ", target_a.shape)

    # check whether the padded images, label and sc_mask have the same shape
    assert img_healthy.shape == img_patho.shape, "image_a and image_b have different shapes"
    assert mask_sc.shape == label_patho.shape, "mask_sc and label have different shapes"

    return img_healthy, mask_sc, img_patho, label_patho


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
    spacing, direction, origin = get_head(path_image_healthy)

    # Load image_healthy and mask_sc
    image_healthy = nib.load(path_image_healthy).get_fdata()
    mask_sc = nib.load(path_mask_sc_healthy).get_fdata()

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
    intensity_ratio = coefficient_of_variation(image_healthy[mask_sc > 0]) / coefficient_of_variation(image_patho[mask_sc_patho > 0])

    # normalize images to range 0 and 1
    image_healthy = (image_healthy - np.min(image_healthy)) / (np.max(image_healthy) - np.min(image_healthy))
    image_patho = (image_patho - np.min(image_patho)) / (np.max(image_patho) - np.min(image_patho))

    # Initialize new_target and new_label with the same shape as target_a
    new_target = np.copy(image_healthy)
    new_label = np.zeros_like(mask_sc)  # because we're creating a new label

    # Check if label_patho has non-zero pixels (ie. there is no lesion in the image)
    if np.count_nonzero(label_patho) == 0:
        print("label_patho has no non-zero pixels")
        return
    # Create 3D bounding box around non-zero pixels in label_patho
    coords = np.argwhere(label_patho > 0)
    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0) + 1  # slices are exclusive at the top
    
    # get the set of all coordinates where SC mask is 1
    new_coords = np.argwhere(mask_sc > 0)

    # make sure that the z-axis is at the max of the SC mask so that it is not mapped on the brainstem
    # TODO: this is basically just removing the top one slice, isnt it?
    # TODO: I updated it to remove 10% of the top slices
    new_coords = new_coords[new_coords[:, 2] < int(new_coords[:, 2].max() * 0.9)]

    # Select random coordinate in new_target where SC mask is 1
    x, y, z = new_coords[random.randint(0, len(new_coords) - 1)]

    # Insert lesion from the bounding box to the new_target
    for x_step, x_cor in enumerate(range(x0, x1)):
        for y_step, y_cor in enumerate(range(y0, y1)):
            for z_step, z_cor in enumerate(range(z0, z1)):
                # Check that dimensions do not overflow
                if x + x_step >= new_target.shape[0] or y + y_step >= new_target.shape[1] or z + z_step >= new_target.shape[2]:
                    continue
                # Insert only voxels corresponding to the lesion mask (label_b)
                # Also make sure that the new lesion is not projected outside of the SC
                if label_patho[x_cor, y_cor, z_cor] > 0 and mask_sc[x + x_step, y + y_step, z + z_step] > 0:
                    new_target[x + x_step, y + y_step, z + z_step] = image_patho[x_cor, y_cor, z_cor] * intensity_ratio
                    new_label[x + x_step, y + y_step, z + z_step] = label_patho[x_cor, y_cor, z_cor]

    # Copy header information from target_a to new_target and new_label
    new_target = copy_head_and_right_xyz(new_target, spacing, direction, origin)
    new_label = copy_head_and_right_xyz(new_label, spacing, direction, origin)

    # Convert i to string and add 3 leading zeros
    s = str(index)
    s = s.zfill(3)

    # Save new_target and new_label
    subject_mame_out = sub_healthy.split('_')[0] + '_' + \
                       sub_patho.split('_')[0] + '_' + \
                       sub_patho.split('_')[1] + '_' + s
    sitk.WriteImage(new_target, os.path.join(args.dir_healthy, subject_mame_out + '_0000.nii.gz'))
    print('Saving new sample: ', os.path.join(args.dir_healthy, subject_mame_out + '_0000.nii.gz'))
    sitk.WriteImage(new_label, os.path.join(args.dir_save, subject_mame_out + '.nii.gz'))
    print('Saving new sample: ', os.path.join(args.dir_save, subject_mame_out + '.nii.gz'))
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
    cases_healthy = simple_cases_healthy[1:]


    """
    Prepare data split, note that validation sets do not participate in 
    CarveMix, and remember to split training sets and validation sets 
    independently in nnunet.training.network_training.nnUNetTrainerV2.do_split 
    when using nnUNet framework
    """
    if args.num > len(cases_patho) or args.num > len(cases_healthy):
        print(f"Number of samples to be generated is larger than the number of available cases. Number of patho cases: "
              f"{len(cases_patho)}, number of healthy cases: {len(cases_healthy)}. Use the smaller of these numbers "
              f"for '-num' argument ")
        exit(1)
    # Get random indices for pathology and healthy subjects
    patho_random_list = np.random.choice(len(cases_patho), args.num, replace=False)
    healthy_random_list = np.random.choice(len(cases_healthy), args.num, replace=False)
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