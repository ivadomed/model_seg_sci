# Automated Segmentation of Spinal Cord and Hyperintense Lesions in Spinal Cord Injury

This repository contains the code for deep learning-based segmentation of the spinal cord and hyperintense lesions in spinal cord injury (SCI). The code is based on the [nnUNet framework](https://github.com/MIC-DKFZ/nnUNet).


## Model Overview

<p align="center" width="100%">
    <img width="95%" src="https://github.com/ivadomed/model_seg_sci/assets/53445351/20e47f31-3d68-4050-bc10-2814f5deb89d">
</p>

## Getting started

### Dependencies

Install Spinal Cord Toolbox. Instructions can be found [here](https://spinalcordtoolbox.com/user_section/installation.html). 

### Step 1: Cloning the Repository

Open a terminal and clone the repository using the following command:

~~~
git clone https://github.com/ivadomed/model_seg_sci.git
~~~

### Step 2: Setting up the Environment

The following commands show how to set up the environment. Note that the documentation assumes that the user has `conda` installed on their system. Instructions on installing `conda` can be found [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

1. Create a conda environment with the following command:
```
conda create -n venv_nnunet python=3.9
```

2. Activate the environment with the following command:
```
conda activate venv_nnunet
```

3. Install the required packages with the following command:
```
pip install -r packaging/requirements.txt
```
 
### Step 3: Getting the Predictions

We provide two methods to run inference on a trained model to obtain the segmentations.

1. **On Individual Images**: This accepts a single image or a list of images. Note that in the case of a list of images, each input image must be separated by a space. Run the following command from the terminal:

```bash
python packaging/run_inference.py --path-images /path/to/image1 /path/to/image2 --path-out /path/to/output --path-model /path/to/model --pred-type {sc-seg, lesion-seg, all}
```

2. On a Dataset: This method performs the inference on all the images in the given dataset. Run the following command from the terminal:

```bash
python packaging/run_inference.py --path-dataset /path/to/test-dataset --path-out /path/to/output --path-model /path/to/model
--pred-type {sc-seg, lesion-seg, all}
```

> **Note**
> The inference scripts also supports inference on a GPU. To do so, simply add the flag `--use-gpu` at the end of the above commands. By default, the inference is run on the CPU. 


