# sci-lesion-segmentation

Preprocessing pipeline to prepare dataset for training lesion segmentation model in SCI.

![Screen Shot 2021-04-28 at 4 15 17 PM](https://user-images.githubusercontent.com/2482071/116466831-f95c1e00-a83c-11eb-9626-d7f668e62d41.png)


## Data organization

Data are organized according to the BIDS structure:

~~~
dataset
├── dataset_description.json
├── participants.json
├── participants.tsv
├── sub-ubc01
├── sub-ubc02
├── sub-ubc03
├── sub-ubc04
├── sub-ubc05
├── sub-ubc06
│   ├── ses-01
│   └── ses-02
|       └── anat
|           ├── sub-ubc06_ses-02_T1w.json
|           ├── sub-ubc06_ses-02_T1w.nii.gz
|           ├── sub-ubc06_ses-02_T2w.json
|           ├── sub-ubc06_ses-02_T2w.nii.gz
|           ├── sub-ubc06_ses-02_acq-ax_T2w.json
|           └── sub-ubc06_ses-02_acq-ax_T2w.nii.gz
|
└── derivatives
    └── labels
        └── sub-ubc06
            └── anat
                ├── sub-ubc06_ses-02_T2w_lesion-manual.json
                └── sub-ubc06_ses-02_T2w_lesion-manual.nii.gz  <---------- manually-created lesion segmentation
~~~

## Data (private)
The data come from the following sites:
- University of Zurich 🇨🇭

