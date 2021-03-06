# Segmentation of T2 hyperintense lesion in acute spinal cord injury

Preprocessing pipeline to prepare dataset for training lesion segmentation model in SCI.

![Screen Shot 2021-04-28 at 4 15 17 PM](https://user-images.githubusercontent.com/2482071/116466831-f95c1e00-a83c-11eb-9626-d7f668e62d41.png)

## Data

Data used for this project hosted on a private repository.

Data for this project come from the following sites (in brackets: the name of the dataset at NeuroPoly's internal server):
- University of Zurich (`sci-zurich`) π¨π­
  - Contrasts: T1w sag, T2w sag, T2w ax
  - Manual segmentation done on the T2w sag
  - Multiple sessions (1, 2, 3)
- University of Colorado Anschutz Medical Campus (`sci-colorado`) πΊπΈ
  - Contrasts: T1w ax, T2w ax
  - Manual segmentation: none
  - Single session

Data are organized according to the [BIDS](https://bids.neuroimaging.io/) structure, as in the example below:

~~~
dataset
βββ dataset_description.json
βββ participants.json
βββ participants.tsv
βββ sub-ubc01
βββ sub-ubc02
βββ sub-ubc03
βββ sub-ubc04
βββ sub-ubc05
βββ sub-ubc06
βΒ Β  βββ ses-01
βΒ Β  βββ ses-02
|    Β Β  βββ anat
|    Β Β   Β Β  βββ sub-ubc06_ses-02_T1w.json
|    Β Β   Β Β  βββ sub-ubc06_ses-02_T1w.nii.gz
|    Β Β   Β Β  βββ sub-ubc06_ses-02_T2w.json
|    Β Β   Β Β  βββ sub-ubc06_ses-02_T2w.nii.gz
|    Β Β   Β Β  βββ sub-ubc06_ses-02_acq-ax_T2w.json
|    Β Β   Β Β  βββ sub-ubc06_ses-02_acq-ax_T2w.nii.gz
|
βββ derivatives
    βββ labels
        βββ sub-ubc06
             Β Β  βββ ses-01
             Β Β  βββ ses-02
                 Β Β  βββ anat
                 Β Β   Β Β  βββ sub-ubc06_ses-02_T2w_seg-manual.json
                 Β Β   Β Β  βββ sub-ubc06_ses-02_T2w_seg-manual.nii.gz  <------------- manually-corrected spinal cord segmentation
                 Β Β   Β Β  βββ sub-ubc06_ses-02_T2w_lesion-manual.json
         Β Β   Β Β          βββ sub-ubc06_ses-02_T2w_lesion-manual.nii.gz  <---------- manually-created lesion segmentation
~~~

More details to convert a dataset into BIDS is available from the [spine-generic](https://spine-generic.readthedocs.io/en/latest/data-acquisition.html#data-conversion-dicom-to-bids) project.

## Getting started

### Dependencies

- [SCT](https://spinalcordtoolbox.com/) commit: 7fd2ea718751dd858840c3823c0830a910d9777c
- [ivadomed](https://ivadomed.org) commit: XXX

### Clone this repository

~~~
git clone https://github.com/ivadomed/model_seg_sci.git
~~~

### Name and Version of the Data

- git@data.neuro.polymtl.ca:datasets/sci-zurich
- Commit: 4ef05bf0b70c04490cd73f433cac4f5f43e5dac3

### Downloading the Dataset
~~~
git clone git@data.neuro.polymtl.ca:datasets/sci-zurich
cd sci-zurich
git annex get .
cd ..
~~~
 
### Prepare the data

The data need to be preprocessed before training. The preprocessing crops the input volume to focus on the region-of-interest i.e. the SC and the lesions. The syntax for preprocessing is:

~~~
sct_run_batch -script preprocessing/preprocess_data.sh -path-data <PATH_TO_DATA>/sci-zurich/ -path-output <PATH_OUTPUT>/sci-zurich-preprocessed -jobs <JOBS>
~~~

where:
- `<JOBS>`: Number of CPU cores to use

### Quality control

After running the preprocessing, it is recommended to check the QC report under `<PATH-OUTPUT>/qc/index.html` and run `preprocessing/qc_preprocess.py` which logs the following statistics as a sanity check: (i) resolutions and sizes for each subject image (both raw and cropped), ii) performs basic shape checks for the cropped SC images and ground-truths (GTs), and most importantly, (iii) checks if any intermediate step during preprocessing (i.e. dilation, cropping) left out any GT lesions.  

TODO: add further details on manual corrections.
TODO: add training details

### Literature

[Here](https://intranet.neuro.polymtl.ca/bibliography/spinal-cord-injury.html#) is a list of relevant articles in relation to this project.

### Things To-Do
1. Current preprocessing deals with multiple sessions within the subjects _independently_ (for simplicity), implying that the sessions are not co-registered and treated as separate subjects. Future versions will incorporate the longitudinal aspect of this, meaning that the sessions will be co-registered with each other before feeding as inputs to the model.
