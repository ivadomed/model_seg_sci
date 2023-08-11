This folder contains the scripts used for creating the initial segmentations of the spinal cord (SC). These were manually corrected and used as labels for training. The manual correction scripts can be found at [this repository](https://github.com/spinalcordtoolbox/manual-correction/). The scripts are used only for `sci-zurich` and `sci-paris` datasets as the manual SC segmentations already existed for the `sci-colorado` dataset.

The `preprocess_zurich_centerline.sh` script was used to create SC centerlines for a few subjects from `sci-zurich` (with heavy metal artefacts) in order to generate the SC segmentations. 

> **Note**
> The scripts were only used for creating the ground-truth labels for model training. No other preprocessing was performed on the data.


### Details on the centerline extraction script

For a high-level overview, the preprocessing script iterates over all the subjects in the dataset and extracts the spinal cord (SC) centerline automatically (if the manual SC centerline is not available), or, uses the manual SC centerline (if available). This centerline mask is then dilated and further used to crop the original image so as to constrain the field-of-view to the close neighbourhood of the SC. 

The flow of `preprocess_zurich_centerline.sh`, in detail, is as follows:

1. A variable corresponding to the contrast to be preprocessed is defined. 
2. The function `get_centerline_if_does_not_exist` checks whether the SC centerline mask exists within the dataset; if yes, uses the manual SC centerline mask, if not, then automatically extracts it using `sct_get_centerline`. The mask is then dilated using the `disk` object. Note that the centerline is extracted using the native resolution of the image. 
3. The original image is resampled to a isotropic resolution. The dilated SC centerline mask is then used as the reference to crop the resampled image so as to focus on the spinal cord.
4. Likewise, the ground truth SC lesion mask is resampled to the same isotropic resolution as the corresponding image and then cropped around the SC using the above-mentioned dilated centerline mask as the reference. 

Note, the cropping is done around the entire region of the SC (so as to reduce size of the input images) and not just around the SC lesion. 

#### Running the Preprocessing Script
The `sct_run_batch` is used to perform the preprocessing on all subjects with following command:
```
sct_run_batch -script preprocess_data.sh -path-data PATH_DATA -path-output PATH_OUT -jobs 64
```

If you intend to run the preprocessing only a few subjects (in this case, 4), then use the following command:
```
sct_run_batch -script preprocess_data.sh -path-data PATH_DATA -path-output PATH_OUT -jobs 64 -include-list sub-zhXX/ses-XX sub-zhXX/ses-XX sub-zhXX/ses-XX sub-zhXX/ses-XX
```
where `PATH_DATA` is the path to the BIDS dataset folder, and `PATH_OUT` is where the output of preprocessing will be saved to. `PATH_OUT` will contain the folders `data_processed/` and `qc/` among others after a successful run.

#### Performing Manual QC for Centerline Extraction
To ensure that the extracted centerline is correct/meaningful, a QC has to be performed by going over all the subjects in the `index.html` file inside the `qc/` folder. The instructions for navigating `index.html` are provided on the page itself. For each subject, two outputs are generated: (1) the extracted centerline as seen on the axial slices and (2) the centerline as seen on the sagittal plane. The QC procedure is as follows:
1. Look at both the outputs and make sure that the centerline is overlayed on the SC itself and does not erroneously lie outside the SC.
2. Identify all the subjects where the automatic centerline detection fails (usually occurs in lumbar images) and download the QC fails. This should download a `.yml` file containing list of subjects required manual correction. 
3. For each failed subject, run the following command from the terminal: `sct_get_centerline -i <subject-name> -method viewer -gap 30 -qc qc-manual -o <output-path>_centerline-manual`. 
    1. This command opens a GUI that allows you to select a few points on the SC so as to extract the centerline. Once you "Save and Quit" the GUI, two files `<subject_name>_centerline-manual.nii.gz` and `<subject_name>_centerline-manual.csv` files are saved in the `output_path` provided.
    2. Note that the `output_path` should be: `derivatives/labels/sub-zhXX/ses-XX/anat/`. That is, the manually corrected SC centerline should exist as a derivative file. 
