#!/bin/bash
#
# Compute tissue bridges using sct_analyze_lesion from GT lesion and spinal cord segmentations (located under
# derivatives/labels)
#
# NOTE: This script requires the following SCT branch:
# https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4489
#
# Usage:
#     sct_run_batch -config config-01a_compute_tissue_bridges_from_GT.json
#
# Example of config-01a_compute_tissue_bridges_from_GT.json:
# {
#  "path_data"   : "<PATH_TO_DATASET>",
#  "path_output" : "<PATH_TO_DATASET>_2024-06-18",
#  "script"      : "<PATH_TO_REPO>/model_seg_sci/tissue_bridges/01a_compute_tissue_bridges_from_GT.sh",
#  "jobs"        : 8,
#  "include_list" : ["sub-zh101", "sub-zh102", "sub-zh103", "sub-zh104", "sub-zh105", "sub-zh106", "sub-zh107", "sub-zh108", "sub-zh109", "sub-zh110", "sub-zh111", "sub-zh112", "sub-zh113", "sub-zh114", "sub-zh115", "sub-zh116", "sub-zh117", "sub-zh118", "sub-zh119"]
# }
#
# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"
#
# Author: Jan Valosek
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

SUBJECT=$1

# ------------------------------------------------------------------------------
# CONVENIENCE FUNCTIONS
# ------------------------------------------------------------------------------

# Copy GT SC or lesion segmentation
copy_gt(){
  local file="$1"
  local type="$2"     # seg or lesion
  # Construct file name to GT SC or lesion segmentation located under derivatives/labels
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file}_${type}-manual.nii.gz"
  echo ""
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
      echo "Found! Copying ..."
      rsync -avzh $FILESEGMANUAL ${file}_${type}-manual.nii.gz
  else
      echo "File ${FILESEGMANUAL}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: Manual GT segmentation ${FILESEGMANUAL}.nii.gz does not exist. Exiting."
      exit 1
  fi
}

# ------------------------------------------------------------------------------
# SCRIPT STARTS HERE
# ------------------------------------------------------------------------------
# get starting time:
start=`date +%s`

# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Copy source T2w images
# Note: we use '/./' in order to include the sub-folder 'ses-0X'
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
# for sci-zurich, copy only sagittal T2w to save space
rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT//[\/]/_}_*sag_T2w.* .

# Go to subject folder for source images
cd ${SUBJECT}/anat

# ------------------------------------------------------------------------------
# T2w
# ------------------------------------------------------------------------------
# sci-zurich
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
file_t2="${SUBJECT//[\/]/_}"_acq-sag_T2w

# Check if file_t2 exists
if [[ ! -e ${file_t2}.nii.gz ]]; then
    echo "File ${file_t2}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
    echo "ERROR: File ${file_t2}.nii.gz does not exist. Exiting."
    exit 1
fi

# ------------------------------------
# GT
# ------------------------------------
# Copy GT SC and lesion segmentations from derivatives/labels
copy_gt "${file_t2}" "seg"
copy_gt "${file_t2}" "lesion"

# Binarize GT lesion segmentation (sct_analyze_lesion requires binary mask until https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4120 is fixed)
sct_maths -i ${file_t2}_lesion-manual.nii.gz -bin 0 -o ${file_t2}_lesion-manual_bin.nii.gz
# Analyze GT SCI lesion segmentation
sct_analyze_lesion -m ${file_t2}_lesion-manual_bin.nii.gz -s ${file_t2}_seg-manual.nii.gz -ofolder ${PATH_RESULTS}

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
