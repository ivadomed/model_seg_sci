import os
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
# NOTE: this file has to be moved to dataset-conversion folder to make it run. the following import only works from there
from utils import Image
import copy
from scipy.ndimage import label
import numpy as np


def keep_largest_object(predictions):
    """Keep the largest connected object from the input array (2D or 3D).

    Taken from:
    https://github.com/ivadomed/ivadomed/blob/1fccf77239985fc3be99161f9eb18c9470d65206/ivadomed/postprocessing.py#L99-L116

    Args:
        predictions (ndarray or nibabel object): Input segmentation. Image could be 2D or 3D.

    Returns:
        ndarray or nibabel (same object as the input).
    """
    # Find number of closed objects using skimage "label"
    labeled_obj, num_obj = label(np.copy(predictions))
    # If more than one object is found, keep the largest one
    if num_obj > 1:
        # Keep the largest object
        predictions[np.where(labeled_obj != (np.bincount(labeled_obj.flat)[1:].argmax() + 1))] = 0
    return predictions


# Function to find all .nii.gz files in a folder
def find_niftis(folder):
    # return sorted(glob.glob(os.path.join(folder, '**', '*.nii.gz'), recursive=True))
    return sorted(glob.glob(os.path.join(folder, '**', '*[0-9][0-9][0-9].nii.gz'), recursive=True))

# Function to create output directory if it doesn't exist
def create_output_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# Function to process a single prediction
def process_prediction(pred_file, output_dir):
    base_name = os.path.basename(pred_file).replace('.nii.gz', '')
    pred_sc = os.path.join(os.path.dirname(pred_file), base_name + '_cord.nii.gz')
    pred_lesion = os.path.join(os.path.dirname(pred_file), base_name + '_lesion.nii.gz')

    # Binarize the prediction file for cord and lesion
    subprocess.run(['sct_maths', '-i', pred_file, '-bin', '0', '-o', pred_sc])
    subprocess.run(['sct_maths', '-i', pred_file, '-bin', '1', '-o', pred_lesion])

    cord_img = Image(pred_sc)

    # keep the largest componment in the lesion
    lesion_img = Image(pred_lesion)
    lesion_img.data = keep_largest_object(lesion_img.data)

    # NOTE: Strangely, combined image with cord>0 = 1 and lesion>0 = 2 is all good in the numpy array
    # but when saving with nib.save(), some float values appear in the image. 
    # as a result, when using np.unique to compute the metrics, it results in values like 0.999912
    # instead of 1 and 2. Hence, converting everything to SCT's Image class which saves images as uint8
    # Convert to SCT's Image class to save the files

    # clone the cord_img class
    combined = copy.deepcopy(cord_img)
    combined.data[lesion_img.data > 0] = 2    

    combined_segmentation = os.path.join(output_dir, f'{base_name}.nii.gz')
    combined.save(combined_segmentation)


def main(prediction_dir, output_dir, num_workers):
    prediction_files = find_niftis(prediction_dir)

    # Create output folder
    create_output_folder(output_dir)

    # Parallel processing of image-label pairs
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        tasks = []
        for prediction_path in prediction_files:
            tasks.append(executor.submit(process_prediction, prediction_path, output_dir))

        # Use tqdm to display a progress bar
        for task in tqdm(tasks):
            task.result()

if __name__ == "__main__":
    import argparse

    # Argument parsing
    parser = argparse.ArgumentParser(description='Post-process predictions to keep the largest connected object.')
    parser.add_argument('--prediction_dir', type=str, help='Directory containing the predictions.')
    parser.add_argument('--output_dir', type=str, help='Output directory to save the processed predictions.')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers to use for parallel processing.')

    args = parser.parse_args()

    main(args.prediction_dir, args.output_dir, args.num_workers)