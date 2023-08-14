import os
import subprocess


def splitext(fname):
    """
    Split a fname (folder/file + ext) into a folder/file and extension.

    Note: for .nii.gz the extension is understandably .nii.gz, not .gz
    (``os.path.splitext()`` would want to do the latter, hence the special case).
    Taken (shamelessly) from: https://github.com/spinalcordtoolbox/manual-correction/blob/main/utils.py
    """
    dir, filename = os.path.split(fname)
    for special_ext in ['.nii.gz', '.tar.gz']:
        if filename.endswith(special_ext):
            stem, ext = filename[:-len(special_ext)], special_ext
            return os.path.join(dir, stem), ext
    # If no special case, behaves like the regular splitext
    stem, ext = os.path.splitext(filename)
    return os.path.join(dir, stem), ext


def add_suffix(fname, suffix):
    """
    Add suffix between end of file name and extension. Taken (shamelessly) from:
    https://github.com/spinalcordtoolbox/manual-correction/blob/main/utils.py

    :param fname: absolute or relative file name. Example: t2.nii.gz
    :param suffix: suffix. Example: _mean
    :return: file name with suffix. Example: t2_mean.nii

    Examples:

    - add_suffix(t2.nii, _mean) -> t2_mean.nii
    - add_suffix(t2.nii.gz, a) -> t2a.nii.gz
    """
    stem, ext = splitext(fname)
    return os.path.join(stem + suffix + ext)


def convert_filenames_to_nnunet_format(path_dataset):
    """
    Convert all filenames in a dataset to the nnUNet format
    :param path_dataset: path to the dataset
    :return: path_dataset: temporary path to the dataset
    """
    # create a temporary folder at the same level as the test folder
    path_tmp = os.path.join(os.path.dirname(path_dataset), 'tmp')
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp, exist_ok=True)

    for f in os.listdir(path_dataset):
        if f.endswith('.nii.gz'):
            # get absolute path to the image
            f = os.path.join(path_dataset, f)
            # add suffix
            f_new = add_suffix(f, '_0000')
            # copy to tmp folder
            os.system('cp {} {}'.format(f, os.path.join(path_tmp, os.path.basename(f_new))))
    
    return path_tmp


def get_orientation(file):
    """
    Get the original orientation of an image
    :param file: path to the image
    :return: orig_orientation: original orientation of the image, e.g. LPI
    """

    # Fetch the original orientation from the output of sct_image
    sct_command = "sct_image -i {} -header | grep -E qform_[xyz] | awk '{{printf \"%s\", substr($2, 1, 1)}}'".format(
        file)
    orig_orientation = subprocess.check_output(sct_command, shell=True).decode('utf-8')

    return orig_orientation


def convert_to_rpi(path_dataset):
    """
    Fetch the original orientation of the images in a dataset, then reorient them to RPI
    :param path_dataset: path to the dataset
    :return: path_dataset: temporary path to the dataset
    :return: orig_orientation_dict: dict of original orientations of the images
    """

    # initialize dict to store original orientations
    orig_orientation_dict = {}

    # iterate through all files, do in-place reorientation to RPI
    for file in os.listdir(path_dataset):
        if file.endswith('.nii.gz'):
            # get absolute path to the image
            fname_file = os.path.join(path_dataset, file)

            # store original orientation for each file
            orig_orientation_dict[file] = get_orientation(fname_file)
            print(f'Original orientation of {file}: {get_orientation(fname_file)}')

        # skip if already in RPI
        if orig_orientation != 'RPI':
            # reorient the image to RPI using SCT
            os.system('sct_image -i {} -setorient RPI -o {}'.format(fname_file, fname_file))

    return path_dataset, orig_orientation_dict


def reorient_to_original_orientation(path_out, orig_orientation_dict):
    """
    Reorient all images in a dataset to the original orientation
    :param path_out: path to the dataset
    :param orig_orientation_dict: dict of original orientations of the images
    :return:
    """
    # iterate through all files, do in-place reorientation to the original orientation
    for file in os.listdir(path_out):
        if file.endswith('.nii.gz'):
            # get absolute path to the image
            fname_file = os.path.join(path_out, file)

            # get original orientation of the image
            orig_orientation = orig_orientation_dict[file]

            # skip if already in RPI
            if orig_orientation != 'RPI':
                # reorient the image to the original orientation using SCT
                os.system('sct_image -i {} -setorient {} -o {}'.format(fname_file, orig_orientation, fname_file))
                print(f'Reorientation to original orientation {orig_orientation} done.')
