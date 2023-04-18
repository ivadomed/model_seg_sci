"""
Take lesion from subject_b and insert it into subject_a

Run:
    python generate_new_lesion.py

nnUNet data structure is required.
TODO: switch to BIDS?
"""

import numpy as np
import random
import SimpleITK as sitk
import nibabel as nib
from tqdm import tqdm
import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_number", "-num", default=5, type=int,
                        help="Number of samples you want to crete: ")
    parser.add_argument("--imagesTr_path", "-imgTr", default="imagesTr", type=str,
                        help="Path to raw images")
    parser.add_argument("--masksTr_path", "-mskTr", default="masksTr", type=str,
                        help="Path to SC masks")
    parser.add_argument("--labelsTr_path", "-labelTr", default="labelsTr", type=str,
                        help="Path to lesion labels")
    parser.add_argument("--mask_check_path", "-mask", default="mask", type=str,
                        help="Path to save masks")
    parser.add_argument("--mixid_csv_path", "-csv", default="mixid_csv.csv", type=str,
                        help="Path to save csv file")
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


def pad_or_crop(target_a, target_b, label_a, label_b):

    # size to match after padding
    if target_a.shape > target_b.shape:
        spatial_size = target_a.shape
        # print("padding image_b to match the dimensions of image_a")

        # the pad function automatically pads the image to the largest dimensions 
        # between the two images. This might result in shape mismatch hence the next step
        # is to crop the padded image to match the size of the other image
        target_b = pad(target_b, spatial_size)
        label_b = pad(label_b, spatial_size)
        # print("paddded image_b: ", target_b.shape)

        if target_b.shape != target_a.shape:
            # crop the padded image until it matches the size of the other image
            if target_b.shape[0] > target_a.shape[0]:
                target_b = target_b[0:target_a.shape[0], :, :]
                label_b = label_b[0:label_a.shape[0], :, :]
            
            if target_b.shape[1] > target_a.shape[1]:
                target_b = target_b[:, 0:target_a.shape[1], :]
                label_b = label_b[:, 0:label_a.shape[1], :]
            
            if target_b.shape[2] > target_a.shape[2]:
                target_b = target_b[:, :,  0:target_a.shape[2]]
                label_b = label_b[:, :, 0:label_a.shape[2]]
            
            # print("padded and cropped image_b: ", target_b.shape)

    else:
        spatial_size = target_b.shape
        # print("padding image_a to match the dimensions of image_b")

        target_a = pad(target_a, spatial_size)
        label_a = pad(label_a, spatial_size)
        # print("paddded image_a: ", target_a.shape)

        if target_b.shape != target_a.shape:
            # crop the padded image until it matches the size of the other image
            if target_a.shape[0] > target_b.shape[0]:
                target_a = target_a[0:target_b.shape[0], :, :]
                label_a = label_a[0:label_b.shape[0], :, :]
            
            if target_a.shape[1] > target_b.shape[1]:
                target_a = target_a[:, 0:target_b.shape[1], :]
                label_a = label_a[:, 0:label_b.shape[1], :]
            
            if target_a.shape[2] > target_b.shape[2]:
                target_a = target_a[:, :,  0:target_b.shape[2]]
                label_a = label_a[:, :, 0:label_b.shape[2]]
            
            # print("padded and cropped image_a: ", target_a.shape)

    return target_a, target_b, label_a, label_b



def generate_new_sample(image_a, image_b, mask_a, mask_b, label_a, label_b):
    spacing, direction, origin = get_head(image_a)

    image_a = nib.load(image_a).get_fdata()
    image_b = nib.load(image_b).get_fdata()
    mask_a = nib.load(mask_a).get_fdata()
    mask_b = nib.load(mask_b).get_fdata()
    label_a = nib.load(label_a).get_fdata()
    label_b = nib.load(label_b).get_fdata()

    # pad and/or crop images and labels so that they have the same shape
    image_a, image_b, label_a, label_b = pad_or_crop(image_a, image_b, label_a, label_b)

    # normalize images
    image_a = (image_a - np.mean(image_a)) / np.std(image_a)
    image_b = (image_b - np.mean(image_b)) / np.std(image_b)

    # Initialize new_target and new_label with the same shape as target_a
    new_target = np.copy(image_a)
    new_label = np.copy(label_a)

    # Create 3D bounding box around non-zero pixels in label_b
    coords = np.argwhere(label_b > 0)
    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0) + 1  # slices are exclusive at the top

    # Select random coordinate in new_target where mask_a is 1
    # TODO - now, random coordinate may be selected close to the edge of the SC --> lesion may be projected outside of
    #  the SC
    new_coords = np.argwhere(mask_a > 0)
    x, y, z = new_coords[random.randint(0, len(new_coords) - 1)]

    # Insert lesion from the bounding box to the new_target
    for x_step, x_cor in enumerate(range(x0, x1)):
        for y_step, y_cor in enumerate(range(y0, y1)):
            for z_step, z_cor in enumerate(range(z0, z1)):
                # Insert only voxels corresponding to the lesion mask (label_b)
                # Also make sure that the new lesion is not projected outside of the SC
                if label_b[x_cor, y_cor, z_cor] > 0 and mask_a[x + x_step, y + y_step, z + z_step] > 0:
                    new_target[x + x_step, y + y_step, z + z_step] = image_b[x_cor, y_cor, z_cor] * ratio
                    new_label[x + x_step, y + y_step, z + z_step] = label_b[x_cor, y_cor, z_cor]

    # Copy header information from target_a to new_target and new_label
    new_target = copy_head_and_right_xyz(new_target, spacing, direction, origin)
    new_label = copy_head_and_right_xyz(new_label, spacing, direction, origin)

    return new_target, new_label


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    if not os.path.exists(args.mask_check_path):
        os.makedirs(args.mask_check_path, exist_ok=True)

    Cases = os.listdir(args.labelsTr_path)
    # remove '.DS_Store' from Cases list
    if '.DS_Store' in Cases:
        Cases.remove('.DS_Store')
    simpleCases = [case.split('.')[0] for i, case in enumerate(Cases) if 'Mix' not in case]
    Cases = simpleCases
    prefix = Cases[0].split('_')[0]

    """
    Prepare data split, note that validation sets do not participate in 
    CarveMix, and remember to split training sets and validation sets 
    independently in nnunet.training.network_training.nnUNetTrainerV2.do_split 
    when using nnUNet framework
    """
    num = len(Cases)
    print('all_set_size: ', num)
    Cases.sort()
    start = 1
    val_num = int(num * 0.2)
    random.seed(985)
    val_id = random.sample(range(1, num - 1), val_num)
    val_id.sort()
    val_set = [Cases[i - start] for i in val_id]
    print('val_set:', val_set)
    print('We use 20% num of imagesTr for validation,\n \
        if you are using demo data file, it will be null')
    tr_set = list(set(Cases) - set(val_set))
    Cases = tr_set
    print('====================================')
    print('=Only select train_set for CarveMix=')
    print('============val_set_not=============')
    print('==========tr_set_size:%d============' % len(Cases))
    print('====================================')
    print(args)

    with open(args.mixid_csv_path, 'w') as f:
        f.write('id,mixid1,mixid2,lam\n')

    """
    Start generating new samples
    """
    for i in tqdm(range(args.generate_number), desc="mixing:"):
        rand_index_a = random.randint(0, len(Cases) - 1)
        rand_index_b = random.randint(0, len(Cases) - 1)

        image_a = os.path.join(args.imagesTr_path, Cases[rand_index_a] + '_0000.nii.gz')
        mask_a = os.path.join(args.masksTr_path, Cases[rand_index_a] + '_mask.nii.gz')
        label_a = os.path.join(args.labelsTr_path, Cases[rand_index_a] + '.nii.gz')
        image_b = os.path.join(args.imagesTr_path, Cases[rand_index_b] + '_0000.nii.gz')
        mask_b = os.path.join(args.masksTr_path, Cases[rand_index_b] + '_mask.nii.gz')
        label_b = os.path.join(args.labelsTr_path, Cases[rand_index_b] + '.nii.gz')

        new_target, new_label = generate_new_sample(image_a, image_b, mask_a, mask_b, label_a, label_b)

        s = str(i)
        sitk.WriteImage(new_target, os.path.join(args.imagesTr_path, prefix + '_CarveMix_' + s + '_0000.nii.gz'))
        sitk.WriteImage(new_label, os.path.join(args.labelsTr_path, prefix + '_CarveMix_' + s + '.nii.gz'))


if __name__ == '__main__':
    main()
