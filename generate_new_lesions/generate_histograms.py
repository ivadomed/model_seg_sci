"""
Generate histograms of the images from the pathology and healthy datasets.

Example:
    python generate_histograms.py
    -dir-pathology
    ~/data/sci-zurich-nnunet/Dataset525_tSCILesionsZurich/imagesTr
    -dir-healthy
    ~/data/data-multi-subject-nnunet/Dataset526_SpineGenericMultiSubject/imagesTr/

"""

import os
import argparse

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from spinalcordtoolbox.image import Image


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir-pathology", default="imagesTr", type=str, required=True,
                        help="Path to raw images from pathology dataset (i.e. SCI-Zurich)")
    parser.add_argument("-dir-healthy", default="imagesTr", type=str, required=True,
                        help="Path to raw images from the healthy dataset (i.e. Spine Generic Multi)")

    return parser


def create_histogram(dir_path, title):

    # get all subjects
    cases = os.listdir(dir_path)
    # remove '.DS_Store' from the list
    if '.DS_Store' in cases:
        cases.remove('.DS_Store')

    # Initialize dictionaries to store histograms for individual subjects
    dict_hist = dict()
    dict_hist_sc = dict()

    # Initialize the figure
    fig, axs = plt.subplots(1, 2, tight_layout=True)
    # Loop across all subjects
    for i, sub in enumerate(cases, 1):

        print(f'Processing subject {i}/{len(cases)}')

        # Whole image
        path_image = os.path.join(dir_path, sub)
        im = Image(path_image)
        # Get numpy array
        im_data = im.data
        # Normalize images to range 0 and 1
        im_data = (im_data - np.min(im_data)) / (np.max(im_data) - np.min(im_data))

        # Spinal cord only
        path_mask = os.path.join(dir_path.replace('imagesTr', 'masksTr'), sub)
        # Remove _0000 from the filename
        path_mask = path_mask.replace('_0000', '')
        im_mask = Image(path_mask)
        # Get numpy array
        im_mask_data = im_mask.data

        # Check if im_data and im_mask_data have the same shape
        if im_data.shape == im_mask_data.shape:

            # Whole image
            # Get histogram using np.histogram
            hist, bin_edges = np.histogram(im_data, bins=50, range=(0, 1))
            # Store histogram in dictionary
            dict_hist[sub] = hist

            # Spinal cord only
            im_data_sc = im_data[im_mask_data > 0]
            # Get histogram using np.histogram
            hist_sc, bin_edges = np.histogram(im_data_sc, bins=50, range=(0, 1))
            # Store histogram in dictionary
            dict_hist_sc[sub] = hist_sc

            # Plot histograms
            axs[0].hist(im_data.flatten(), bins=50, histtype='step', range=(0, 1))
            axs[0].set_title('Whole image')
            axs[1].hist(im_data_sc.flatten(), bins=50, histtype='step', range=(0, 1))
            axs[1].set_title('Spinal cord only')
        else:
            print(f'Skipping subject {sub} because image and SC mask have different shapes.')

    # Compute mean histogram across subjects
    hist_mean = pd.DataFrame.from_dict(dict_hist, orient='index').mean()
    hist_sc_mean = pd.DataFrame.from_dict(dict_hist_sc, orient='index').mean()

    # # Plot using sns.distplot
    # sns.displot(pd.DataFrame.from_dict(dict_hist, orient='index'), bins=50, kind="kde")
    # sns.distplot(hist_mean, bins=50, ax=axs[0])
    # sns.distplot(hist_sc_mean, bins=50, ax=axs[1])

    axs[0].plot(bin_edges[:-1], hist_mean, label='Mean histogram', color='red', linewidth=2)
    axs[1].plot(bin_edges[:-1], hist_sc_mean, label='Mean histogram', color='red', linewidth=2)

    # Add legend
    axs[0].legend()
    axs[1].legend()

    # Add master title
    fig.suptitle(title)

    # Save figure
    plt.savefig('histogram_' + title + '.png')
    print('Histogram saved as histogram_' + title + '.png')
    # Close figure
    plt.close()


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Expand user (i.e. ~) in paths
    args.dir_healthy = os.path.expanduser(args.dir_healthy)
    args.dir_pathology = os.path.expanduser(args.dir_pathology)

    create_histogram(args.dir_pathology, 'patho')
    create_histogram(args.dir_healthy, 'healthy')


if __name__ == '__main__':
    main()
