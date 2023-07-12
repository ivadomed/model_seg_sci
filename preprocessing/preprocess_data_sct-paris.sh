#!/bin/bash
#
# Segment spinal cord and generate QC.
#
# Dependencies (versions):
# - SCT (5.8)
#
# Usage:
# sct_run_batch -script preprocessing/preprocess_data_sct-paris.sh -path-data <DATA> -path-output <DATA>_2023-XX-XX -jobs 24

# Manual segmentations or labels should be located under:
# PATH_DATA/derivatives/labels/SUBJECT/ses-0X/anat/

# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"
#
# Authors: Jan Valosek
#

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Print retrieved variables from sct_run_batch to the log (to allow easier debug)
echo "Retrieved variables from from the caller sct_run_batch:"
echo "PATH_DATA: ${PATH_DATA}"
echo "PATH_DATA_PROCESSED: ${PATH_DATA_PROCESSED}"
echo "PATH_RESULTS: ${PATH_RESULTS}"
echo "PATH_LOG: ${PATH_LOG}"
echo "PATH_QC: ${PATH_QC}"

# -------------------------------------------------------------------------
# CONVENIENCE FUNCTIONS
# -------------------------------------------------------------------------

segment_if_does_not_exist() {
  ###
  #  This function checks if a manual spinal cord segmentation file already exists, then:
  #    - If it does, copy it locally.
  #    - If it doesn't, perform automatic spinal cord segmentation
  #  This allows you to add manual segmentations on a subject-by-subject basis without disrupting the pipeline.
  ###
  local file="$1"
  local contrast="$2"
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
    # Segment spinal cord
    sct_deepseg_sc -i ${file}.nii.gz -c ${contrast} -qc ${PATH_QC} -qc-subject ${SUBJECT}

  fi
}

copy_lesion() {
  ###
  #  This function checks if a manual lesion segmentation exists, then:
  #    - If it does, copy it locally
  ###
  local image="$1"
  # Update global variable with segmentation file name
  FILELESION="${image}_lesion"
  FILELESIONMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILELESION}-manual.nii.gz"
  echo "Looking for manual lesion segmentation: $FILELESIONMANUAL"
  if [[ -e $FILELESIONMANUAL ]]; then
    echo "Found! Using manual lesion segmentation."
    rsync -avzh $FILELESIONMANUAL ${FILELESION}.nii.gz

    # Make sure the lesion is binary
    sct_maths -i ${FILELESION}.nii.gz -bin 0 -o ${FILELESION}_bin.nii.gz
  fi

}

# Retrieve input params and other params
SUBJECT=$1

# get starting time:
start=`date +%s`

# -------------------------------------------------------------------------
# SCRIPT STARTS HERE
# -------------------------------------------------------------------------
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Copy source images
# Note: we use '/./' in order to include the sub-folder 'ses-0X'
rsync -Ravzh $PATH_DATA/./$SUBJECT .

# Go to subject folder for source images
cd ${SUBJECT}/anat

# Define variables
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
file="${SUBJECT//[\/]/_}"

# -------------------------------------------------------------------------
# T2w sag
# -------------------------------------------------------------------------
# Add suffix corresponding to contrast
file_t2w=${file}_T2w
# Check if T2w image exists
if [[ -f ${file_t2w}.nii.gz ]];then

    # Spinal cord segmentation
    segment_if_does_not_exist ${file_t2w} 't2'

    # Automatic SC seg fails at the lesion, so we add the lesion to the segmentation (to save time with manual corrections)
    # Copy SCI lesion
    copy_lesion ${file_t2w}
    # Add lesion to SC segmentation
    sct_maths -i ${file_t2w}_seg.nii.gz -add ${FILELESION}_bin.nii.gz -o ${file_t2w}_seg.nii.gz
    # Make sure the final SC segmentation is binary
    sct_maths -i ${file_t2w}_seg.nii.gz -bin 0.5 -o ${file_t2w}_seg.nii.gz
    sct_qc -i ${file_t2w}.nii.gz -p sct_deepseg_sc -s ${file_t2w}_seg.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}

fi

# -------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------

# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"