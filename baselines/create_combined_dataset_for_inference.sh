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
    echo "Usage: $0 <json_file> <zurich_folder> <colorado_folder> <output_folder>"
    exit 1
fi

JSON_FILE=$1
ZURICH_FOLDER=$2
COLORADO_FOLDER=$3
OUTPUT_FOLDER=$4

mkdir -p $OUTPUT_FOLDER/derivatives/labels/

# Loop across subjects in test_subjects_sci_seed50.json
for subject in $(cat ${JSON_FILE} | jq -r 'keys[]'); do
    echo $subject
    # sci-zurich
    if [[ $subject =~ "sub-zh" ]]; then
        cp -r $ZURICH_FOLDER/$subject $OUTPUT_FOLDER/
        cp -r $ZURICH_FOLDER/derivatives/labels/$subject $OUTPUT_FOLDER/derivatives/labels/
        echo "Copied $subject from sci-zurich to $OUTPUT_FOLDER"
    else
        # sci-colorado
        cp -r $COLORADO_FOLDER/$subject $OUTPUT_FOLDER/
        cp -r $COLORADO_FOLDER/derivatives/labels/$subject $OUTPUT_FOLDER/derivatives/labels/
        echo "Copied $subject from sci-colorado $OUTPUT_FOLDER"
    fi
done
