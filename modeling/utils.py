import numpy as np
import torch


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

