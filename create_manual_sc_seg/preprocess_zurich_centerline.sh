#!/bin/bash
#
# Preprocess data.
#
# Dependencies (versions):
# - SCT (5.4)
#
# Usage:
# sct_run_batch -script preprocess_data.sh -path-data <PATH-TO-DATASET> -path-output <PATH-TO-OUTPUT> -jobs -1

# Manual segmentations or labels should be located under:
# PATH_DATA/derivatives/labels/SUBJECT/ses-0X/anat/

# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"

# Global variables
# CENTERLINE_METHOD="svm"  # method sct_deepseg_sc uses for centerline extraction: 'svm', 'cnn'
CENTERLINE_EXTRACTION_METHOD="optic"  # method sct_get_centerline uses: 'optic', 'viewer' and 'fitseg'

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Print retrieved variables from the sct_run_batch script to the log (to allow easier debug)
echo "Retrieved variables from the caller sct_run_batch:"
echo "PATH_DATA: ${PATH_DATA}"
echo "PATH_DATA_PROCESSED: ${PATH_DATA_PROCESSED}"
echo "PATH_RESULTS: ${PATH_RESULTS}"
echo "PATH_LOG: ${PATH_LOG}"
echo "PATH_QC: ${PATH_QC}"

# CONVENIENCE FUNCTIONS
# ======================================================================================================================

get_centerline_if_does_not_exist() {
  ###
  #  This function checks if a manual spinal cord centerline file already exists, then:
  #    - If it does, copies it locally.
  #    - If it doesn't, performs automatic spinal cord centerline extraction.
  #  This allows you to add manual centerline files on a subject-by-subject basis without disrupting the pipeline.
  ###
  local file="$1"
  local contrast="$2"
  local centerline_extraction_method="$3"
  # Update global variable with the centerline file name
  FILECENTERLINE="${file}_centerline"
  FILECENTERLINEMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILECENTERLINE}-manual.nii.gz"
  echo
  echo "Looking for manual centerline: $FILECENTERLINEMANUAL"
  if [[ -e $FILECENTERLINEMANUAL ]]; then
    echo "Found! Using manual centerline file."
    rsync -avzh $FILECENTERLINEMANUAL ${FILECENTERLINE}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILECENTERLINE}.nii.gz -p sct_get_centerline -qc ${PATH_QC} -qc-subject ${SUBJECT}
    # Generate sagittal QC (using sct_label_utils) to check that the centerline is correct on the sagittal plane
    # NOTE: sct_label_vertebrae is raising an error when using manual centerline, so we use sct_label_utils instead
    sct_qc -i ${file}.nii.gz -s ${FILECENTERLINE}.nii.gz -p sct_label_utils -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic centerline extraction."
    # Extract spinal cord centerline based on the specified centerline method
    if [[ $centerline_extraction_method == "optic" ]]; then
      sct_get_centerline -i ${file}.nii.gz -c $contrast -method optic -qc ${PATH_QC} -qc-subject ${SUBJECT} 
      # Generate sagittal QC (using sct_label_vertebrae) to check that the centerline is correct on the sagittal plane
      sct_qc -i ${file}.nii.gz -s ${FILECENTERLINE}.nii.gz -p sct_label_vertebrae -qc ${PATH_QC} -qc-subject ${SUBJECT}
    # elif [[ $centerline_method == "svm" ]]; then
    #   sct_deepseg_sc -i ${file}.nii.gz -c $contrast -centerline svm -qc ${PATH_QC} -qc-subject ${SUBJECT}
    else
      echo "Centerline extraction method = ${centerline_method} is not recognized!"
      exit 1
    fi
  fi
}

# Retrieve input params and other params
SUBJECT=$1

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

# Copy segmentation ground truths (GT)
mkdir -p derivatives/labels
rsync -Ravzh $PATH_DATA/derivatives/labels/./$SUBJECT derivatives/labels/.

# Go to subject folder for source images
cd ${SUBJECT}/anat

# Define variables
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
file="${SUBJECT//[\/]/_}"

# Add suffix corresponding to contrast
file=${file}_acq-sag_T2w

# Make sure the image metadata is a valid JSON object
if [[ ! -s ${file}.json ]]; then
  echo "{}" >> ${file}.json
fi

# Spinal cord centerline extraction using the T2w contrast
get_centerline_if_does_not_exist ${file} t2 ${CENTERLINE_EXTRACTION_METHOD}

# define variable for naming the centerline output
file_centerline="${FILECENTERLINE}"

# Dilate the spinal cord (SC) centerline mask
sct_maths -i ${file_centerline}.nii.gz -dilate 15 -shape disk -dim 1 -o ${file_centerline}_dilate.nii.gz

# Define the absolute path for dilated centerline mask because the directory is changed later
fname_centerline_dilate=${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${file_centerline}_dilate

# Resample the image using linear interpolation
sct_resample -i ${file}.nii.gz -mm 1x1x1 -x linear -o ${file}_resample.nii.gz

# Resample the dilated centerline mask as well
sct_resample -i ${fname_centerline_dilate}.nii.gz -mm 1x1x1 -x linear -o ${fname_centerline_dilate}_resample.nii.gz

# Use dilated SC centerline mask to crop the resampled image
sct_crop_image -i ${file}_resample.nii.gz -m ${fname_centerline_dilate}_resample.nii.gz -o ${file}_crop.nii.gz

# Go to subject folder for segmentation GTs
cd $PATH_DATA_PROCESSED/derivatives/labels/$SUBJECT/anat

# Define variables
file_gt="${file}_lesion-manual"

# Make sure the GT metadata is a valid JSON object
if [[ ! -s ${file_gt}.json ]]; then
  echo "{}" >> ${file_gt}.json
fi

# Resample the GT using linear interpolation
sct_resample -i ${file_gt}.nii.gz -mm 1x1x1 -x linear -o ${file_gt}_resample.nii.gz

# Crop the GT image using resampled and dilated SC centerline mask
sct_crop_image -i ${file_gt}_resample.nii.gz -m ${fname_centerline_dilate}_resample.nii.gz -o ${file_gt}_crop.nii.gz

# Go back to the root output path
cd $PATH_OUTPUT

# Create and populate clean data processed folder for training
PATH_DATA_PROCESSED_CLEAN="${PATH_DATA_PROCESSED}_clean"

# Copy over required BIDs files
mkdir -p $PATH_DATA_PROCESSED_CLEAN $PATH_DATA_PROCESSED_CLEAN/${SUBJECT} $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/anat
rsync -avzh $PATH_DATA_PROCESSED/dataset_description.json $PATH_DATA_PROCESSED_CLEAN/
rsync -avzh $PATH_DATA_PROCESSED/participants.* $PATH_DATA_PROCESSED_CLEAN/
rsync -avzh $PATH_DATA_PROCESSED/README $PATH_DATA_PROCESSED_CLEAN/
rsync -avzh $PATH_DATA_PROCESSED/dataset_description.json $PATH_DATA_PROCESSED_CLEAN/derivatives/

# For lesion segmentation task, copy SC crops as inputs and lesion annotations as targets
# NOTE: we are not cropping the .json files as they are corrupted, thereby resulting in errors while using ivadomed
rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/anat/${file}_crop.nii.gz $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/anat/${file}.nii.gz
# rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/anat/${file}.json $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/anat/${file}.json
mkdir -p $PATH_DATA_PROCESSED_CLEAN/derivatives $PATH_DATA_PROCESSED_CLEAN/derivatives/labels $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT} $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/anat/
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels/${SUBJECT}/anat/${file_gt}_crop.nii.gz $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/anat/${file_gt}.nii.gz
# rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels/${SUBJECT}/anat/${file_gt}.json $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/anat/${file_gt}.json


# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"