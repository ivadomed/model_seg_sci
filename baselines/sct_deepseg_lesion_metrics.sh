#!/bin/bash
#
# Preprocess data.
#
# Dependencies (versions):
# - SCT (6.3.0)
#
# Usage:
# sct_run_batch -script preprocess_data.sh -path-data <PATH-TO-DATASET> -path-output <PATH-TO-OUTPUT> -jobs <num-cpu-cores>

# Manual segmentations or labels should be located under:
# PATH_DATA/derivatives/labels/SUBJECT/ses-0X/anat/

# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"

# Global variables
CENTERLINE_METHOD="svm"  # method sct_deepseg_sc uses for centerline extraction: 'svm', 'cnn'

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Retrieve input params and other params
SUBJECT=$1

# Check if straightened or not
if [[ $2 == "straightened" ]]; then
  STRAIGHTENED=True
else
  STRAIGHTENED=False
fi

echo "STRAIGHTENED is set to $STRAIGHTENED"

# get starting time:
start=`date +%s`

# SCRIPT STARTS HERE
# ==============================================================================
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Copy BIDS-required files to processed data folder (e.g. list of participants)
if [[ ! -f "participants.tsv" ]]; then
  rsync -avzh $PATH_DATA/participants.tsv .
fi
if [[ ! -f "participants.json" ]]; then
  rsync -avzh $PATH_DATA/participants.json .
fi
if [[ ! -f "dataset_description.json" ]]; then
  rsync -avzh $PATH_DATA/dataset_description.json .
fi
if [[ ! -f "README" ]]; then
  rsync -avzh $PATH_DATA/README .
fi

# Copy source images
# Note: we use '/./' in order to include the sub-folder 'ses-0X'
rsync -Ravzh $PATH_DATA/./$SUBJECT .

# I am deliberately as I want perserve the structure of the data
# Copy segmentation ground truths (GT)
# mkdir -p derivatives/labels
# rsync -Ravzh $PATH_DATA/derivatives/labels/./$SUBJECT . # derivatives/labels/.

# print the current contents of the PATH_DATA_PROCESSED folder
echo "The contents of the PATH_DATA_PROCESSED folder are:"
ls -l

# Define variables
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
# file="${SUBJECT//[\/]/_}"

# Copy BIDS-required files to processed data folder
#rsync -avzh $PATH_DATA/{participants.tsv,participants.json,dataset_description.json,README} .

# Loop over all axial files in the subject's anatomy directory

cd ${SUBJECT}/anat
shopt -s nullglob # Handle the case where no files match the pattern
counter=1  # Initialize counter at 1

for file in $(find "$PWD" -type f -name "*acq-ax*" ! -name "*seg*" ! -name "*lesion*" -name "*.nii.gz" | sort); do
    echo "Processing $file"

    path_base=$(echo "$file" | sed 's/\.nii\.gz$//')
    file_name=$(basename "$file" ".nii.gz")

    # Make sure the image metadata is a valid JSON object
    if [[ ! -s ${path_base}.json ]]; then
      echo "{}" > ${path_base}.json
    fi

    sct_image -i ${file} -set-sform-to-qform -o ${file}
    sct_image -i ${file} -setorient RPI -o ${file}
    sct_deepseg_lesion -i ${file} -ofolder . -c t2_ax -centerline $CENTERLINE_METHOD -brain 0

    mkdir -p ${PATH_DATA_PROCESSED}/derivatives/labels/${SUBJECT}/anat/

    if [[ "$STRAIGHTENED" == True ]]; then

      # Use sed to replace desc-straightened with lesion-manual_desc-straightened
      modified_file_name=$(echo "$file_name" | sed 's/desc-straightened/lesion-manual_desc-straightened/')
      rsync -avzh ${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${modified_file_name}.nii.gz ${PATH_DATA_PROCESSED}/derivatives/labels/${SUBJECT}/anat/${modified_file_name}.nii.gz
      rsync -avzh ${path_base}_lesionseg.nii.gz ${PATH_DATA_PROCESSED}/derivatives/labels/${SUBJECT}/anat/${file_name}_seg-deepseglesion_desc-straightened.nii.gz
      sct_qc -i ${file} -s ${PATH_DATA_PROCESSED}/derivatives/labels/${SUBJECT}/anat/${file_name}_seg-deepseglesion_desc-straightened.nii.gz \
      -d ${PATH_DATA_PROCESSED}/derivatives/labels/${SUBJECT}/anat/${file_name}_seg-deepseglesion_desc-straightened.nii.gz \
      -p sct_deepseg_lesion -qc ${PATH_QC} -qc-subject ${SUBJECT} -plane axial

      python3 /home/$(whoami)/git_repositories/MetricsReloaded/compute_metrics_reloaded.py \
      -reference  ${PATH_DATA_PROCESSED}/derivatives/labels/${SUBJECT}/anat/${modified_file_name}.nii.gz \
      -prediction ${PATH_DATA_PROCESSED}/derivatives/labels/${SUBJECT}/anat/${file_name}_seg-deepseglesion_desc-straightened.nii.gz \
      -output ${PATH_DATA_PROCESSED}/derivatives/labels/${SUBJECT}/anat/${file_name}_metrics-deepseglesion_desc-straightened.csv \
      -metrics dsc nsd vol_diff rel_vol_error lesion_ppv lesion_sensitivity lesion_f1_score

    else
      rsync -avzh ${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file_name}_lesion-manual.nii.gz ${PATH_DATA_PROCESSED}/derivatives/labels/${SUBJECT}/anat/${file_name}_lesion-manual.nii.gz
      rsync -avzh ${path_base}_lesionseg.nii.gz ${PATH_DATA_PROCESSED}/derivatives/labels/${SUBJECT}/anat/${file_name}_seg-deepseglesion.nii.gz
      sct_qc -i ${file} -s ${PATH_DATA_PROCESSED}/derivatives/labels/${SUBJECT}/anat/${file_name}_seg-deepseglesion.nii.gz \
      -d ${PATH_DATA_PROCESSED}/derivatives/labels/${SUBJECT}/anat/${file_name}_seg-deepseglesion.nii.gz \
      -p sct_deepseg_lesion -qc ${PATH_QC} -qc-subject ${SUBJECT} -plane axial

      python3 /home/$(whoami)/git_repositories/MetricsReloaded/compute_metrics_reloaded.py \
      -reference  ${PATH_DATA_PROCESSED}/derivatives/labels/${SUBJECT}/anat/${file_name}_lesion-manual.nii.gz \
      -prediction ${PATH_DATA_PROCESSED}/derivatives/labels/${SUBJECT}/anat/${file_name}_seg-deepseglesion.nii.gz \
      -output ${PATH_DATA_PROCESSED}/derivatives/labels/${SUBJECT}/anat/${file_name}_metrics-deepseglesion.csv \
      -metrics dsc nsd vol_diff rel_vol_error lesion_ppv lesion_sensitivity lesion_f1_score

    fi

done

shopt -u nullglob # Turn off nullglob to return to normal glob behavior

# Go back to the root output path
cd $PATH_OUTPUT

# Create and populate clean data processed folder for training
# PATH_DATA_PROCESSED_CLEAN="${PATH_DATA_PROCESSED}_clean"

# Copy over required BIDs files
# mkdir -p $PATH_DATA_PROCESSED_CLEAN $PATH_DATA_PROCESSED_CLEAN/${SUBJECT} $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/anat
# rsync -avzh $PATH_DATA_PROCESSED/dataset_description.json $PATH_DATA_PROCESSED_CLEAN/
# rsync -avzh $PATH_DATA_PROCESSED/participants.* $PATH_DATA_PROCESSED_CLEAN/
# rsync -avzh $PATH_DATA_PROCESSED/README $PATH_DATA_PROCESSED_CLEAN/

# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"