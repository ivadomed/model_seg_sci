import numpy as np
from scipy import ndimage
from skimage import measure


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


def get_lesion_volume(im_patho_lesion_data, voxel_dims, debug=False):
    # Compute volume
    nonzero_voxel_count = np.count_nonzero(im_patho_lesion_data)
    voxel_size = np.prod(voxel_dims)
    nonzero_voxel_volume = nonzero_voxel_count * voxel_size

    if debug:
        print("Voxel size = {}".format(voxel_size))
        print("Number of non-zero voxels = {}".format(nonzero_voxel_count))
        print(f"Volume of non-zero voxels = {nonzero_voxel_volume:.2f} mm^3")

    return nonzero_voxel_volume


def keep_largest_component(new_lesion_data):
    """
    Keep only the largest connected component in the lesion mask
    """
    # Get connected components
    labels = measure.label(new_lesion_data)
    # Get number of connected components
    num_components = labels.max()
    # Get size of each connected component
    component_sizes = np.bincount(labels.ravel())
    # Get largest connected component
    largest_component = np.argmax(component_sizes[1:]) + 1
    # Keep only the largest connected component
    new_lesion_data[labels != largest_component] = 0

    return new_lesion_data
