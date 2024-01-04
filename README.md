# Automated Segmentation of Spinal Cord and Hyperintense Lesions in Traumatic Spinal Cord Injury

[![medRxiv](https://img.shields.io/badge/medRxiv-10.1101/2024.01.03.24300794v1-blue.svg)](https://www.medrxiv.org/content/10.1101/2024.01.03.24300794v1)

This repository contains the code for deep learning-based segmentation of the spinal cord and hyperintense lesions in spinal cord injury (SCI). The code is based on the [nnUNet framework](https://github.com/MIC-DKFZ/nnUNet).


## Model Overview

The model was trained on raw T2-weighted images of SCI patients from multiple sites. The data included images with both axial and sagittal resolutions. To ensure uniformity across sites, all images were initially re-oriented to RPI. Given an input image, the model is able to segment *both* the lesion and the spinal cord. 

<img width="1000" alt="sciseg-overview" src="https://github.com/ivadomed/model_seg_sci/assets/53445351/f463e0d2-8c2e-42e7-bb2c-5122962ca373">

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

ℹ️ To temporarily suppress warnings raised by the nnUNet, you can run the following three commands in the same terminal session as the above command:

```bash
export nnUNet_raw="${HOME}/nnUNet_raw"
export nnUNet_preprocessed="${HOME}/nnUNet_preprocessed"
export nnUNet_results="${HOME}/nnUNet_results"
```

To segment a single image using the trained model, run the following command from the terminal. This assumes that the model has been downloaded and is available locally.

For lesion segmentation:

```bash
python packaging/run_inference_single_subject.py -i sub-001_T2w.nii.gz -o sub-001_T2w_lesion_seg_nnunet.nii.gz -path-model /path/to/model -pred-type lesion
```

For spinal cord segmentation:

```bash
python packaging/run_inference_single_subject.py -i sub-001_T2w.nii.gz -o sub-001_T2w_seg_nnunet.nii.gz -path-model /path/to/model -pred-type sc
```

For segmenting a dataset of multiple subjects (instead of a single subject), run the following command from the 
terminal.

```bash
python packaging/run_inference.py --path-dataset /path/to/test-dataset --path-out /path/to/output-directory --path-model /path/to/model --pred-type {sc-seg, lesion-seg, all}
```

ℹ️ The script also supports getting segmentations on a GPU. To do so, simply add the flag `--use-gpu` at the end of the above commands. By default, the inference is run on the CPU. It is useful to note that obtaining the predictions from the GPU is significantly faster than the CPU.

## Citation Info

If you find this work and/or code useful for your research, please cite our paper:

```
@article {Enamundram2024.01.03.24300794,
	author = {Naga Karthik Enamundram* and Jan Valosek* and Andrew C Smith and Dario Pfyffer and Simon Schading-Sassenhausen and Lynn Farner and Kenneth Arnold Weber II and Patrick Freund and Julien Cohen-Adad},
	title = {SCIseg: Automatic Segmentation of T2-weighted Hyperintense Lesions in Spinal Cord Injury},
	elocation-id = {2024.01.03.24300794},
	year = {2024},
	doi = {10.1101/2024.01.03.24300794},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2024/01/03/2024.01.03.24300794},
	eprint = {https://www.medrxiv.org/content/early/2024/01/03/2024.01.03.24300794.full.pdf},
	journal = {medRxiv},
    note = {*Shared first authorship}
}

```
