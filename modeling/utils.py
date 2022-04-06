import numpy as np
import torch
import torchvision.utils as vutils
import wandb
from collections import defaultdict

def volume2subvolumes(volume, subvolume_size, stride_size):
    """Converts 3D volumes into 3D subvolumes; works with PyTorch tensors."""
    subvolumes = []
    assert volume.ndim == 3

    for x in range(0, (volume.shape[0] - subvolume_size[0])+1, stride_size[0]):
        for y in range(0, (volume.shape[1] - subvolume_size[1])+1, stride_size[1]):
            for z in range(0, (volume.shape[2] - subvolume_size[2])+1, stride_size[2]):
                subvolumes.append(
                    volume[
                        x: (x+subvolume_size[0]),
                        y: (y+subvolume_size[1]),
                        z: (z+subvolume_size[2])
                    ])
    return subvolumes

def subvolumes2volume(subvolumes, volume_size):
    """Converts list of 3D subvolumes into 3D volumes; works with Numpy arrays."""
    volume = np.zeros(volume_size)
    subvolume_size = subvolumes[0].shape
    num_sbv_per_dim = [volume_size[i] // subvolume_size[i] for i in range(3)]

    for i, x in enumerate(range(0, (volume_size[0]-subvolume_size[0])+1, subvolume_size[0])):
        for j, y in enumerate(range(0, (volume_size[1]-subvolume_size[1])+1, subvolume_size[1])):
            for k, z in enumerate(range(0, (volume_size[2]-subvolume_size[2])+1, subvolume_size[2])):
                # indices get multiplied with the number of subvolumes remaining in the next dimension(s)
                subvolume_index = i*np.prod(num_sbv_per_dim[1:]) + j*num_sbv_per_dim[2] + k
                volume[
                    x: (x+subvolume_size[0]),
                    y: (y+subvolume_size[1]),
                    z: (z+subvolume_size[2])
                ] = subvolumes[subvolume_index]
    
    return volume

def convert_labels_to_RGB(grid_img):
    """Converts 2D images to RGB encoded images for display on WandB.
    Taken from https://github.com/ivadomed/ivadomed/blob/master/ivadomed/visualize.py#107
    Args: grid_img (Tensor): GT or prediction tensor with dimensions (batch size, number of classes, height, width).
    Returns: tensor: RGB image with shape (height, width, 3).
    """
    # Keep always the same color labels
    batch_size, n_class, h, w = grid_img.shape
    rgb_img = torch.zeros((batch_size, 3, h, w))

    # Keep always the same color labels
    np.random.seed(6)
    for i in range(n_class):
        r, g, b = np.random.randint(0, 256, size=3)
        rgb_img[:, 0, ] = r * grid_img[:, i, ]
        rgb_img[:, 1, ] = g * grid_img[:, i, ]
        rgb_img[:, 2, ] = b * grid_img[:, i, ]

    return rgb_img

def save_wandb_img(dataset_type, input_samples, gt_samples, preds):
    """Saves input images, gt and predictions in WandB.
    Taken from: https://github.com/ivadomed/ivadomed/blob/master/ivadomed/visualize.py#L131
    Args:
        dataset_type (str): Choice between Training or Validation.
        input_samples (Tensor): Input images with shape (batch size, number of channel, height, width, depth) if 3D else
            (batch size, number of channel, height, width)
        gt_samples (Tensor): GT images with shape (batch size, number of channel, height, width, depth) if 3D else
            (batch size, number of channel, height, width)
        preds (Tensor): Model's prediction with shape (batch size, number of channel, height, width, depth) if 3D else
            (batch size, number of channel, height, width)
    """
    # Take all images stacked on depth dimension
    num_2d_img = input_samples.shape[-1]
    input_samples_copy = input_samples.clone()
    preds_copy = preds.clone()
    gt_samples_copy = gt_samples.clone()
    for idx in range(num_2d_img):
        input_samples = input_samples_copy[..., idx]
        preds = preds_copy[..., idx]
        gt_samples = gt_samples_copy[..., idx]
        # Only display images with labels
        if gt_samples.sum() == 0:
            continue

        # take only one modality for grid
        if not isinstance(input_samples, list) and input_samples.shape[1] > 1:
            tensor = input_samples[:, 0, ][:, None, ]
            input_samples = torch.cat((tensor, tensor, tensor), 1)
        elif isinstance(input_samples, list):
            input_samples = input_samples[0]

        grid_img = vutils.make_grid(torch.transpose(input_samples, 2, 3), normalize=True, scale_each=True)
        wandb.log({dataset_type+"/Input": wandb.Image(grid_img)})

        grid_img = vutils.make_grid(torch.transpose(convert_labels_to_RGB(preds), 2, 3), normalize=True, scale_each=True)
        wandb.log({dataset_type+"/Predictions": wandb.Image(grid_img)})

        grid_img = vutils.make_grid(torch.transpose(convert_labels_to_RGB(gt_samples), 2, 3), normalize=True, scale_each=True)
        wandb.log({dataset_type+"/Ground-Truth": wandb.Image(grid_img)})


class MetricManager(object):
    """Computes specified metrics and stores them in a dictionary.
    Args:
        metric_fns (list): List of metric functions.
    Attributes:
        metric_fns (list): List of metric functions.
        result_dict (dict): Dictionary storing metrics.
        num_samples (int): Number of samples.
    """

    def __init__(self, metric_fns):
        self.metric_fns = metric_fns
        self.num_samples = 0
        self.result_dict = defaultdict(list)

    def __call__(self, prediction, ground_truth):
        self.num_samples += len(prediction)
        for metric_fn in self.metric_fns:
            for p, gt in zip(prediction, ground_truth):
                res = metric_fn(p, gt)
                dict_key = metric_fn.__name__
                self.result_dict[dict_key].append(res)

    def get_results(self):
        res_dict = {}
        for key, val in self.result_dict.items():
            if np.all(np.isnan(val)):  # if all values are np.nan
                res_dict[key] = None
            else:
                res_dict[key] = np.nanmean(val)
        return res_dict

    def reset(self):
        self.num_samples = 0
        self.result_dict = defaultdict(list)    