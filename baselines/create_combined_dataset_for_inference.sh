#!/bin/bash
#
# Combine subjects from sci-zurich and sci-colorado datasets into a single BIDS-like folder, example output:
#
# ├── derivatives
# │	 └── labels
# │	     ├── sub-5416   # sci-colorado subject
# │	     │	 └── anat
# │	     │	     ├── sub-5416_T2w_lesion-manual.json
# │	     │	     ├── sub-5416_T2w_lesion-manual.nii.gz
# │	     │	     ├── sub-5416_T2w_seg-manual.json
# │	     │	     └── sub-5416_T2w_seg-manual.nii.gz
# │	     └── sub-zh01   # sci-zurich subject
# │	         └── ses-01
# │	             └── anat
# │	                 ├── sub-zh01_ses-01_acq-sag_T2w_lesion-manual.json
# │	                 ├── sub-zh01_ses-01_acq-sag_T2w_lesion-manual.nii.gz
# │	                 ├── sub-zh01_ses-01_acq-sag_T2w_seg-manual.json
# │	                 └── sub-zh01_ses-01_acq-sag_T2w_seg-manual.nii.gz
# ├── sub-5416    # sci-colorado subject
# │	 └── anat
# │	     ├── sub-5416_T2w.json
# │	     └── sub-5416_T2w.nii.gz
# └── sub-zh01    # sci-zurich subject
#    └── ses-01
#        └── anat
#            ├── sub-zh01_ses-01_acq-sag_T2w.json
#            └── sub-zh01_ses-01_acq-sag_T2w.nii.gz
#
# Author: Jan Valosek
#

# Exit if no input arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <yaml_file> <zurich_folder> <colorado_folder> <output_folder>"
    exit 1
fi

YAML_FILE=$1        # YAML file with train and test subjects, e.g., dataset_split_seed123.yaml
ZURICH_FOLDER=$2    # Path to the sci-zurich BIDS dataset
COLORADO_FOLDER=$3  # Path to the sci-colorado BIDS dataset
OUTPUT_FOLDER=$4    # Path to the output folder where the combined dataset will be stored

# Create the output folder and derivatives/labels subfolder
mkdir -p $OUTPUT_FOLDER/derivatives/labels/

# Retrieve test participant IDs (e.g., `sub-6577` or `sub-zh63`) from the provided YAML file
TEST_SUBJECTS=$(python -c "import yaml; print('\n'.join([item.split('_')[0] for item in yaml.safe_load(open('${YAML_FILE}'))['test']]))")

# Loop across test subjects
for subject in ${TEST_SUBJECTS}; do
    echo "Processing: $subject"
    # sci-zurich
    if [[ $subject =~ "sub-zh" ]]; then
        cp -r $ZURICH_FOLDER/$subject $OUTPUT_FOLDER/
        cp -r $ZURICH_FOLDER/derivatives/labels/$subject $OUTPUT_FOLDER/derivatives/labels/
        echo "Copied $subject from $ZURICH_FOLDER to $OUTPUT_FOLDER"
    # sci-colorado
    else
        cp -r $COLORADO_FOLDER/$subject $OUTPUT_FOLDER/
        cp -r $COLORADO_FOLDER/derivatives/labels/$subject $OUTPUT_FOLDER/derivatives/labels/
        echo "Copied $subject from $COLORADO_FOLDER $OUTPUT_FOLDER"
    fi
done
