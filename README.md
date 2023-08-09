# Automated Segmentation of the Spinal Cord and Hyperintense Lesions in Acute Spinal Cord Injury

This repository contains the code for deep learning-based segmentation of the spinal cord and T2-weighted hyperintense lesions in acute spinal cord injury (SCI). The code is based on the well-known [nnUNet framework](https://github.com/MIC-DKFZ/nnUNet).


## Model Overview

TODO: add a screenshot/gif of input, network and output


## Data

The data used for training the model are hosted on a private repository according to the [BIDS](https://bids.neuroimaging.io) standard. They are gathered from three different sites as shown below (in brackets: the name of the dataset at NeuroPoly's internal server):

- University of Zurich (`sci-zurich`) (N=) 🇨🇭
  - Contrasts available: Sagittal T1w, Sagittal T2w, Axial T2w
  - Contrasts used for training: Sagittal T2w
  - Manual segmentations for both spinal cord (SC) and lesion only available for Sagittal T2w
  - Mix of single and multiple (up to 3) sessions
  - Number of subjects: 97
- University of Colorado Anschutz Medical Campus (`sci-colorado`) 🇺🇸
  - Contrasts available: Axial T1w, Axial T2w
  - Contrasts used for training: Axial T2w
  - Manual segmentations for both SC and lesion only available for Axial T2w
  - Single session
  - Number of subjects: 80
- XXX (`sci-paris`) 🇫🇷
  - Contrasts available: Axial T2w
  - Contrasts used for training: Axial T2w
  - Manual segmentations for both SC and lesion available for Axial T2w
  - Single session
  - Number of subjects: 14


## Getting started

### Dependencies

TODO: add updated requirements.txt --> those are the dependencies

### Step 1: Cloning the Repository

~~~
git clone https://github.com/ivadomed/model_seg_sci.git
~~~

### Names and Versions of the Dataset

- `sci-zurich`
  - Name: git@data.neuro.polymtl.ca:datasets/sci-zurich
  - Commit: b3cb6f51  (can be checked by running `git log --oneline` after downloading the dataset)
- `sci-colorado`
  - Name: git@data.neuro.polymtl.ca:datasets/sci-colorado
  - Commit: 1518ecd
- `sci-paris`
  - Name: git@data.neuro.polymtl.ca:datasets/sci-paris
  - Commit: c4e3bf7

### Downloading the Dataset

The downloading procedure is same for all the three datasets, just replace the "<dataset-name>" by the name of the dataset you want to download from the server.

~~~
git clone git@data.neuro.polymtl.ca:datasets/<dataset-name>
cd <dataset-name>
git annex get .
cd ..
~~~
 
### Step 2: Preparing the Data

The best part about this model is that there is **no preprocessing required**! The model is directly trained on the raw data. The only data preparation step is to convert the data to the nnUNet format. The following commands are used for converting the dataset. 

TODOs: 
1. add SCT's RPI conversion 
2. add commands for dataset conversion


### Quality control

After running the preprocessing, it is recommended to check the QC report under `<PATH-OUTPUT>/qc/index.html` and run `preprocessing/qc_preprocess.py` which logs the following statistics as a sanity check: (i) resolutions and sizes for each subject image (both raw and cropped), ii) performs basic shape checks for the cropped SC images and ground-truths (GTs), and most importantly, (iii) checks if any intermediate step during preprocessing (i.e. dilation, cropping) left out any GT lesions.  

TODO: add further details on manual corrections.
TODO: add training details

### Literature

[Here](https://intranet.neuro.polymtl.ca/bibliography/spinal-cord-injury.html#) is a list of relevant articles in relation to this project.

### Things To-Do
1. Current preprocessing deals with multiple sessions within the subjects _independently_ (for simplicity), implying that the sessions are not co-registered and treated as separate subjects. Future versions will incorporate the longitudinal aspect of this, meaning that the sessions will be co-registered with each other before feeding as inputs to the model.

### Inference

The instructions for running inference can be found [here]().


