#!/bin/bash
#
# Preprocess data.
#
# Dependencies (versions):
# - SCT (5.4.0)
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


# CONVENIENCE FUNCTIONS
# ======================================================================================================================

segment_if_does_not_exist() {
  ###
  #  This function checks if a manual spinal cord segmentation file already exists, then:
  #    - If it does, copy it locally.
  #    - If it doesn't, perform automatic spinal cord segmentation.
  #  This allows you to add manual segmentations on a subject-by-subject basis without disrupting the pipeline.
  ###
  local file="$1"
  local contrast="$2"
  local centerline_method="$3"
  # Update global variable with segmentation file name
  FILESEG="${file}_seg"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILESEG}-manual.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic segmentation."
    # Segment spinal cord based on the specified centerline method
    if [[ $centerline_method == "cnn" ]]; then
      sct_deepseg_sc -i ${file}.nii.gz -c $contrast -brain 1 -centerline cnn -qc ${PATH_QC} -qc-subject ${SUBJECT}
    elif [[ $centerline_method == "svm" ]]; then
      sct_deepseg_sc -i ${file}.nii.gz -c $contrast -centerline svm -qc ${PATH_QC} -qc-subject ${SUBJECT}
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

# Add suffix corresponding to contrast (Sagittal T2w)
file_sag=${file}_acq-sag_T2w

# Add suffix corresponding to contrast (Axial T2w)
file_ax=${file}_acq-ax_T2w

# Make sure the Sagittal image metadata is a valid JSON object
if [[ ! -s ${file_sag}.json ]]; then
  echo "{}" >> ${file_sag}.json
fi

# Make sure the Axial image metadata is a valid JSON object
if [[ ! -s ${file_ax}.json ]]; then
  echo "{}" >> ${file_ax}.json
fi

# Resample the Sagittal T2w to isotropic 1mm x 1mm x 1mm resolution
sct_resample -i ${file_sag}.nii.gz -mm 1x1x1 -o ${file_sag}_res.nii.gz

# Bring the Axial image to the same space as the "resampled" Sagittal image (for visualization purposes only)
sct_register_multimodal -i ${file_ax}.nii.gz -d ${file_sag}_res.nii.gz -o ${file_ax}_same.nii.gz -identity 1

# Spinal cord segmentation using the Sagittal T2w contrast
segment_if_does_not_exist ${file_sag}_res t2 ${CENTERLINE_METHOD}
file_sag_res_seg="${FILESEG}"

# Spinal cord segmentation using the Axial T2w contrast
segment_if_does_not_exist ${file_ax} t2 ${CENTERLINE_METHOD}
file_ax_seg="${FILESEG}"

# Perform the registration axial T2w (moving) --> sag T2w (fixed)
# NOTE: We are passing the un-resampled / original axial image as the input to avoid double interpolation
sct_register_multimodal -i ${file_ax}.nii.gz -iseg ${file_ax_seg}.nii.gz -d ${file_sag}_res.nii.gz -dseg ${file_sag_res_seg}.nii.gz -o ${file_ax}_reg.nii.gz -param step=1,type=seg,algo=slicereg,metric=MeanSquares,smooth=3

# Dilate spinal cord masks for both images 
sct_maths -i ${file_sag_res_seg}.nii.gz -dilate 5 -shape ball -o ${file_sag_res_seg}_dilate.nii.gz
sct_maths -i ${file_ax_seg}.nii.gz -dilate 5 -shape ball -o ${file_ax_seg}_dilate.nii.gz

# TODO: Also crop the axial image ?
# Use dilated mask to crop the resampled image and manual MS segmentations
sct_crop_image -i ${file_sag}_res.nii.gz -m ${file_sag_res_seg}_dilate.nii.gz -o ${file_sag}_crop.nii.gz


# Go to subject folder for segmentation GTs
cd $PATH_DATA_PROCESSED/derivatives/labels/$SUBJECT/anat

# Define variables
file_sag_gt="${file_sag}_lesion-manual"

# Redefine variable for final SC segmentation mask as path changed
file_sag_res_seg_dil=${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${file_sag_res_seg}_dilate

# Make sure the first rater metadata is a valid JSON object
if [[ ! -s ${file_sag_gt}.json ]]; then
  echo "{}" >> ${file_sag_gt}.json
fi

# Resample GT to isotropic 1mm x 1mm x 1mm resolution
sct_resample -i ${file_sag_gt}.nii.gz -mm 1x1x1 -o ${file_sag_gt}_res.nii.gz

# Crop the manual seg
sct_crop_image -i ${file_sag_gt}_res.nii.gz -m ${file_sag_res_seg_dil}.nii.gz -o ${file_sag_gt}_crop.nii.gz

# Go back to the root output path
cd $PATH_OUTPUT

# # Create and populate clean data processed folder for training
# PATH_DATA_PROCESSED_CLEAN="${PATH_DATA_PROCESSED}_clean"

# # Copy over required BIDs files
# mkdir -p $PATH_DATA_PROCESSED_CLEAN $PATH_DATA_PROCESSED_CLEAN/${SUBJECT} $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/anat
# rsync -avzh $PATH_DATA_PROCESSED/dataset_description.json $PATH_DATA_PROCESSED_CLEAN/
# rsync -avzh $PATH_DATA_PROCESSED/participants.* $PATH_DATA_PROCESSED_CLEAN/
# rsync -avzh $PATH_DATA_PROCESSED/README $PATH_DATA_PROCESSED_CLEAN/
# rsync -avzh $PATH_DATA_PROCESSED/dataset_description.json $PATH_DATA_PROCESSED_CLEAN/derivatives/

# # For lesion segmentation task, copy SC crops as inputs and lesion annotations as targets
# rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/anat/${file_sag}_crop.nii.gz $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/anat/${file_sag}.nii.gz
# rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/anat/${file_sag}.json $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/anat/${file_sag}.json
# mkdir -p $PATH_DATA_PROCESSED_CLEAN/derivatives $PATH_DATA_PROCESSED_CLEAN/derivatives/labels $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT} $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/anat/
# rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels/${SUBJECT}/anat/${file_sag_gt}_crop.nii.gz $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/anat/${file_sag_gt}.nii.gz
# rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels/${SUBJECT}/anat/${file_sag_gt}.json $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/anat/${file_sag_gt}.json


# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
