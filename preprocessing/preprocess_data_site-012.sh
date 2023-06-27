#!/bin/bash
#
# Process site-012 dataset from spineimage.ca
#
# Usage:
#     sct_run_batch -script preprocessing/preprocess_data_site-012.sh -path-data <PATH_TO_DATA> -path-output <PATH_TO_DATA>_2023-06-27 -jobs 24 -exclude sub-hal022
#
# Note: sub-hal022 has two runs; both are severely corrupted by artifacts --> the subject is skipped
#
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

# Print retrieved variables from the sct_run_batch script to the log (to allow easier debug)
echo "Retrieved variables from from the caller sct_run_batch:"
echo "PATH_DATA: ${PATH_DATA}"
echo "PATH_DATA_PROCESSED: ${PATH_DATA_PROCESSED}"
echo "PATH_RESULTS: ${PATH_RESULTS}"
echo "PATH_LOG: ${PATH_LOG}"
echo "PATH_QC: ${PATH_QC}"

# CONVENIENCE FUNCTIONS
# ======================================================================================================================
# Check if manual spinal cord segmentation file already exists. If it does, copy it locally.
# If it doesn't, perform automatic spinal cord segmentation
segment_if_does_not_exist() {
  local file="$1"
  local contrast="$2"
  # Update global variable with segmentation file name
  FILESEG="${file}_label-SC_mask"
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
    # sub-hal008 has highly curved spinal cord --> use init centerline (created manually)
    if [[ ${SUBJECT} == "sub-hal008" ]]; then
        sct_deepseg_sc -i ${file}.nii.gz -o ${FILESEG}.nii.gz -c ${contrast} -centerline file -file_centerline ${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file}_centerline.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}
    else
        sct_deepseg_sc -i ${file}.nii.gz -o ${FILESEG}.nii.gz -c ${contrast} -qc ${PATH_QC} -qc-subject ${SUBJECT}
    fi
  fi
}

# Retrieve input params and other params
SUBJECT=$1

# get starting time:
start=`date +%s`

# ------------------------------------------------------------------------------
# SCRIPT STARTS HERE
# ------------------------------------------------------------------------------
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Copy source T2w sag images
# Note: we use '/./' in order to include the sub-folder 'ses-0X'
# sub-hal003 has multiple images covering the whole spine --> copy only the first one covering cervical spine
if [[ ${SUBJECT} == "sub-hal003" ]]; then
    rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT}_*acq-sag_run-01_T2w* .
    mv sub-hal003/anat/sub-hal003_acq-sag_run-01_T2w.nii.gz sub-hal003/anat/sub-hal003_acq-sag_T2w.nii.gz
    mv sub-hal003/anat/sub-hal003_acq-sag_run-01_T2w.json sub-hal003/anat/sub-hal003_acq-sag_T2w.json
# sub-hal004 has two runs --> copy only the first one (the second one is a repeat and corrupted by artifacts)
elif [[ ${SUBJECT} == "sub-hal004" ]]; then
    rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT}_*acq-sag_run-01_T2w* .
    mv sub-hal004/anat/sub-hal004_acq-sag_run-01_T2w.nii.gz sub-hal004/anat/sub-hal004_acq-sag_T2w.nii.gz
    mv sub-hal004/anat/sub-hal004_acq-sag_run-01_T2w.json sub-hal004/anat/sub-hal004_acq-sag_T2w.json
# sub-hal006 has two runs --> copy only the second one (the first one contains slight artifacts)
elif [[ ${SUBJECT} == "sub-hal006" ]]; then
    rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT}_*acq-sag_run-02_T2w* .
    mv sub-hal006/anat/sub-hal006_acq-sag_run-02_T2w.nii.gz sub-hal006/anat/sub-hal006_acq-sag_T2w.nii.gz
    mv sub-hal006/anat/sub-hal006_acq-sag_run-02_T2w.json sub-hal006/anat/sub-hal006_acq-sag_T2w.json
# other subjects have only one run --> copy it
else
    rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT}_*acq-sag_T2w* .
fi

# Go to subject folder for source images
cd ${SUBJECT}/anat

# ------------------------------------------------------------------------------
# T2w Sagittal
# ------------------------------------------------------------------------------
# Define variables
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
file_t2_sag="${SUBJECT//[\/]/_}"_acq-sag_T2w
# Check if file_t2_sag exists
if [[ ! -e ${file_t2_sag}.nii.gz ]]; then
    echo "File ${file_t2_sag}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
    echo "ERROR: File ${file_t2_sag}.nii.gz does not exist. Exiting."
    exit 1
else
    # Segment SC
    segment_if_does_not_exist ${file_t2_sag} 't2'
    file_t2_sag_seg=$FILESEG
fi

# ------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------
# Display results (to easily compare integrity across SCT versions)
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
