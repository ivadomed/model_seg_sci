# sci-lesion-segmentation
Preprocessing pipeline to prepare dataset for training lesion segmentation model in SCI.

## Data organization

Data are organized according to the BIDS structure:

~~~
    dataset
    │
    ├── dataset_description.json
    ├── participants.json
    ├── participants.tsv
    ├── sub-ubc01
    ├── sub-ubc02
    ├── sub-ubc03
    ├── sub-ubc04
    ├── sub-ubc05
    ├── sub-ubc06
    │   │
    │   ├── ses-01
    │   ├── ses-02
    |   │   ├── anat
    |   │   │   ├── sub-ubc06_ses-02_T1w.json
    |   │   │   ├── sub-ubc06_ses-02_T1w.nii.gz
    |   │   │   ├── sub-ubc06_ses-02_T2w.json
    |   │   │   ├── sub-ubc06_ses-02_T2w.nii.gz
    |   │   │   ├── sub-ubc06_ses-02_acq-ax_T2w.json
    |   │   │   ├── sub-ubc06_ses-02_acq-ax_T2w.nii.gz
    |
    └── derivatives
        │
        └── labels
            └── sub-ubc06
                │
                ├── anat
                │   ├── sub-ubc06_ses-02_T2w_lesion-manual.json
                │   ├── sub-ubc06_ses-02_T2w_lesion-manual.nii.gz  <---------- manually-created lesion segmentation
~~~
