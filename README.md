# Automated Segmentation of Spinal Cord and Hyperintense Lesions in Spinal Cord Injury

This repository contains the code for deep learning-based segmentation of the spinal cord and hyperintense lesions in spinal cord injury (SCI). The code is based on the [nnUNet framework](https://github.com/MIC-DKFZ/nnUNet).


## Model Overview

The model was trained on raw T2-weighted images of SCI patients from multiple sites. The data included images with both axial and sagittal resolutions. To ensure uniformity across sites, all images were initially re-oriented to RPI. Given an input image, the model is able to segment *both* the lesion and the spinal cord. 

<p align="center" width="100%">
    <img width="95%" src="https://github.com/ivadomed/model_seg_sci/assets/53445351/38cf629c-52a4-4894-a9bb-b9afa834d320">
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
cd model_seg_sci
pip install -r packaging/requirements.txt
```
 
### Step 3: Getting the Predictions

To segment the image(s) using the trained model, run the following command from the terminal. This assumes that the model has been downloaded and is available locally.

```bash
python packaging/run_inference.py --path-dataset /path/to/test-dataset --path-out /path/to/output-directory --path-model /path/to/model --pred-type {sc-seg, lesion-seg, all}
```

> **Note**
> The script also supports getting segmentations on a GPU. To do so, simply add the flag `--use-gpu` at the end of the above commands. By default, the inference is run on the CPU. 
> It is useful to note that obtaining the predictions from the GPU is significantly faster than the CPU.


