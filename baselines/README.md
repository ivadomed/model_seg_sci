
Compare our nnUNet model with other methods (`sct_propseg`, `sct_deepseg_sc -kernel 2d`, `sct_deepseg_sc -kernel 3d`) on 
`sci-zurich` and `sci-colorado` datasets.

## Data structure

Subjects from both datasets have to be located in the same BIDS-like folder, example:

```
├── derivatives
│	 └── labels
│	     ├── sub-5416     # sci-colorado subject
│	     │	 └── anat
│	     │	     ├── sub-5416_T2w_lesion-manual.json
│	     │	     ├── sub-5416_T2w_lesion-manual.nii.gz
│	     │	     ├── sub-5416_T2w_seg-manual.json
│	     │	     └── sub-5416_T2w_seg-manual.nii.gz
│	     └── sub-zh01     # sci-zurich subject
│	         └── ses-01
│	             └── anat
│	                 ├── sub-zh01_ses-01_acq-sag_T2w_lesion-manual.json
│	                 ├── sub-zh01_ses-01_acq-sag_T2w_lesion-manual.nii.gz
│	                 ├── sub-zh01_ses-01_acq-sag_T2w_seg-manual.json
│	                 └── sub-zh01_ses-01_acq-sag_T2w_seg-manual.nii.gz
├── sub-5416      # sci-colorado subject
│	 └── anat
│	     ├── sub-5416_T2w.json
│	     └── sub-5416_T2w.nii.gz
└── sub-zh01      # sci-zurich subject
 └── ses-01
     └── anat
         ├── sub-zh01_ses-01_acq-sag_T2w.json
         └── sub-zh01_ses-01_acq-sag_T2w.nii.gz
```

## Dependencies

### nnUNet

`conda` environment with nnUNetV2 is required to run this script. See installation instructions [here](https://github.com/ivadomed/utilities/blob/main/quick_start_guides/nnU-Net_quick_start_guide.md#installation).

### ANIMA

ANIMA is used to compute segmentation performance metrics. See installation instructions [here](https://github.com/ivadomed/utilities/blob/main/quick_start_guides/ANIMA_quick_start_guide.md).

### SCT

Follow installation instructions [here](https://github.com/spinalcordtoolbox/spinalcordtoolbox#installation).

## Running the script

```bash
sct_run_batch -config config.json
```

Example of the `config.json` file:
```json
 {
  "path_data"   : "<PATH_TO_DATASET>",
  "path_output" : "<PATH_TO_DATASET>_2023-08-18",
  "script"      : "<PATH_TO_REPO>/model_seg_sci/baselines/comparison_with_other_methods.sh",
  "jobs"        : 8,
  "script_args" : "<PATH_TO_REPO>/model_seg_sci/baselines/run_inference_single_subject.py <PATH_TO_MODEL>/sci-multisite-model"
 }
```