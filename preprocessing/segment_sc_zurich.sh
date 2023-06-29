#!/bin/bash
#
# Segment spinal cord for sci-zurich dataset
#
# Usage:
#     sct_run_batch -script segment_sc_zurich.sh -path-data <PATH-TO-DATASET> -path-output <PATH-TO-OUTPUT> -jobs <num-cpu-cores>
#
# Manual labels are located under /derivatives/labels directory.
#
# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"
#
# Authors: Jan Valosek, Naga Karthik
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

# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

segment_if_does_not_exist() {
  ###
  #  This function checks if a manual spinal cord segmentation file already exists, then:
  #    - If it does, copy it locally.
  #    - If it doesn't, the function checks if a manual centerline exists:
  #       - If it does, copy it locally and use it as input centerline for segmentation.
  #       - If it doesn't, perform automatic spinal cord segmentation without centerline.
  #  This allows you to add manual segmentations on a subject-by-subject basis without disrupting the pipeline.
  ###
  local file="$1"
  local contrast="$2"
  # Update global variable with segmentation file name
  FILESEG="${file}_seg"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILESEG}-manual.nii.gz"
  FILECENTERLINE="${file}_centerline"
  FILECENTERLINEMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILECENTERLINE}-manual.nii.gz"

  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
      echo "Manual segmentation not found."
      echo "Looking for manual centerline: $FILECENTERLINEMANUAL"
      if [[ -e $FILECENTERLINEMANUAL ]]; then
        echo "Found! Using manual centerline file for segmentation."
        rsync -avzh $FILECENTERLINEMANUAL ${FILECENTERLINE}.nii.gz
        sct_deepseg_sc -i ${file}.nii.gz -c $contrast -centerline file -file_centerline ${FILECENTERLINE}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}
      else
        # Segment spinal cord
        echo "Not found. Proceeding with automatic segmentation without centerline."
        sct_deepseg_sc -i ${file}.nii.gz -c $contrast -qc ${PATH_QC} -qc-subject ${SUBJECT}
      fi
  fi
}

# Retrieve input params and other params
SUBJECT=$1

# get starting time:
start=`date +%s`

# ==============================================================================
# SCRIPT STARTS HERE
# ==============================================================================

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

# Add suffix corresponding to contrast
file=${file}_acq-sag_T2w

# Spinal cord segmentation using the T2w contrast
segment_if_does_not_exist ${file} t2 ${CENTERLINE_METHOD}

# ==============================================================================
# END
# ==============================================================================

# Display results (to easily compare integrity across SCT versions)
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
