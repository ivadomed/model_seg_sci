from scipy import ndimage
import numpy as np

def get_bounding_boxes(mask):
    """
    Adapted from: https://ivadomed.org/_modules/ivadomed/object_detection/utils.html#get_bounding_boxes 
    Generates a 3D bounding box around a given mask.
    Args:
        mask (Numpy array): Mask of the ROI.

    Returns:
        list: Bounding box coordinate (x_min, x_max, y_min, y_max, z_min, z_max).
    """

    # Label the different objects in the mask
    labeled_mask, _ = ndimage.label(mask)
    object_labels = np.unique(labeled_mask)
    bounding_boxes = []
    for label in object_labels[1:]:
        single_object = labeled_mask == label
        coords = np.where(single_object)
        dimensions = []
        for i in range(len(coords)):
            dimensions.append(int(coords[i].min())-1)
            dimensions.append(int(coords[i].max())+1)
        bounding_boxes.append(dimensions)

    return bounding_boxes