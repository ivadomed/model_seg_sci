import os
import re
import numpy as np

import matplotlib.pyplot as plt

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


def fetch_subject_and_session(filename_path):
    """
    Get subject ID, session ID and filename from the input BIDS-compatible filename or file path
    The function works both on absolute file path as well as filename
    More about BIDS - https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#anatomy-imaging-data
    :param filename_path: input nifti filename (e.g., sub-001_ses-01_T1w.nii.gz) or file path
    (e.g., /home/user/MRI/bids/sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz
    :return: subjectID: subject ID (e.g., sub-001)
    :return: sessionID: session ID (e.g., ses-01)
    :return: filename: nii filename (e.g., sub-001_ses-01_T1w.nii.gz)
    """

    _, filename = os.path.split(filename_path)              # Get just the filename (i.e., remove the path)
    subject = re.search('sub-(.*?)[_/]', filename_path)     # [_/] slash or underscore
    subjectID = subject.group(0)[:-1] if subject else None    # [:-1] removes the last underscore or slash
    session = re.findall(r'ses-..', filename_path)
    sessionID = session[0] if session else None               # Return None if there is no session
    # REGEX explanation
    # \d - digit
    # \d? - no or one occurrence of digit
    # *? - match the previous element as few times as possible (zero or more times)
    # . - any character

    return subjectID, sessionID, filename


def generate_histogram(im_healthy_data, im_healthy_sc_data,
                       im_patho_data, im_patho_sc_data, im_patho_sc_dil_data, im_patho_lesion_data,
                       im_augmented_data, im_augmented_lesion_data, new_sc_data,
                       sub_healthy, sub_patho, subject_name_out,
                       output_dir):
    """
    Generate healthy-patho pair histogram for the whole image and for their SCs
    :param im_healthy_data: healthy image data
    :param im_healthy_sc_data: healthy image SC data
    :param im_patho_data: patho image data
    :param im_patho_sc_data: patho image SC data
    :param im_patho_lesion_data: patho image lesion data
    :param figure_path: path to save the figure
    """

    # Check if directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    figure_path = output_dir + f"/{subject_name_out}_histogram.png"

    # Create 1x2 subplots
    fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(15, 5))
    # Whole images
    axs[0].hist(im_healthy_data.flatten(), bins=50, range=(0, 1), label=f'Healthy subject ({sub_healthy})',
                alpha=0.3, histtype='step', linewidth=3, color='green')
    axs[0].hist(im_patho_data.flatten(), bins=50, range=(0, 1), label=f'Patho subject ({sub_patho})',
                alpha=0.3, histtype='step', linewidth=3, color='red')
    axs[0].hist(im_augmented_data.flatten(), bins=50, range=(0, 1), label=f'Augmented subject ({subject_name_out})',
                alpha=0.3, histtype='step', linewidth=3, color='blue')
    axs[0].set_title('Whole image')

    # Spinal cords only
    # Healthy SC
    axs[1].hist(im_healthy_data[im_healthy_sc_data > 0].flatten(), bins=50, range=(0, 1),
                label=f'Healthy SC ({sub_healthy})', alpha=0.3, histtype='step', linewidth=3, color='green')
    # Patho SC minus lesion
    axs[1].hist(im_patho_data[(im_patho_sc_data > 0) & (im_patho_lesion_data == 0)].flatten(), bins=50, range=(0, 1),
                label=f'Patho SC ({sub_patho})', alpha=0.3, histtype='step', linewidth=3, color='red')
    # Patho SC dilated minus lesion
    axs[1].hist(im_patho_data[(im_patho_sc_dil_data > 0) & (im_patho_lesion_data == 0)].flatten(), bins=50, range=(0, 1),
                label=f'Patho SC dilated ({sub_patho})', alpha=0.9, histtype='step', linewidth=3, color='pink')
    # Augmented SC
    axs[1].hist(im_augmented_data[(new_sc_data > 0) & (im_augmented_lesion_data == 0)].flatten(), bins=50, range=(0, 1),
                label=f'Augmented SC ({subject_name_out})', alpha=0.3, histtype='step', linewidth=3, color='blue')
    # Lesion only
    axs[1].hist(im_patho_data[im_patho_lesion_data > 0].flatten(), bins=50, range=(0, 1),
                label=f'Lesion ({sub_patho})', alpha=0.6, histtype='step', linewidth=3, color='orange')
    # Augmented lesion only
    axs[1].hist(im_augmented_data[im_augmented_lesion_data > 0].flatten(), bins=50, range=(0, 1),
                label=f'Augmented lesion ({subject_name_out})', alpha=0.9, histtype='step', linewidth=3, color='lightblue')
    axs[1].set_title('Spinal cord only')

    # Add legend to top right corner and decrease font size
    axs[0].legend(loc='upper right', prop={'size': 8})
    axs[1].legend(loc='upper right', prop={'size': 8})
    # Add x labels
    axs[0].set_xlabel('Normalized Intensity')
    axs[1].set_xlabel('Normalized Intensity')
    # Add y labels
    axs[0].set_ylabel('Count')
    axs[1].set_ylabel('Count')
    # Save plot
    plt.savefig(figure_path, dpi=300)
    print(f"Saved histogram to {figure_path}")
    # Close plot
    plt.close(fig)
