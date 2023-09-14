#!/bin/bash
#
# The script crops the input image, spinal cord GT, and lesion GT around the spinal cord segmentation (based on the
# dilated spinal cord GT)
#
# Usage:
#     sct_run_batch -config config.json
#
# Example of config.json:
# {
#  "path_data"   : "<PATH_TO_DATASET>",
#  "path_output" : "<PATH_TO_DATASET>_2023-09-14",
#  "script"      : "<PATH_TO_REPO>/model_seg_sci/dataset-conversion/crop_around_sc.sh",
#  "jobs"        : 8
# }
#
# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"
#
# Author: Jan Valosek, Naga Karthik
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

echo "SUBJECT: ${SUBJECT}"


# CONVENIENCE FUNCTIONS
# ======================================================================================================================
# Copy GT spinal cord segmentation
copy_gt_sc(){
  local file="$1"
  # Construct file name to GT segmentation located under derivatives/labels
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file}_seg-manual.nii.gz"
  echo ""
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
      echo "Found! Copying ..."
      rsync -avzh $FILESEGMANUAL ${file}_seg-manual.nii.gz
  else
      echo "File ${FILESEGMANUAL}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: Manual GT segmentation ${FILESEGMANUAL}.nii.gz does not exist. Exiting."
      exit 1
  fi
}

# Copy GT lesion segmentation
copy_gt_lesion(){
  local file="$1"
  # Construct file name to GT segmentation located under derivatives/labels
  FILELESIONMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file}_lesion-manual.nii.gz"
  echo ""
  echo "Looking for manual segmentation: $FILELESIONMANUAL"
  if [[ -e $FILELESIONMANUAL ]]; then
      echo "Found! Copying ..."
      rsync -avzh $FILELESIONMANUAL ${file}_lesion-manual.nii.gz
  else
      echo "File ${FILELESIONMANUAL}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: Manual GT segmentation ${FILELESIONMANUAL}.nii.gz does not exist. Exiting."
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
if [[ $SUBJECT =~ "sub-zh" ]]; then
  # for sci-zurich, copy only sagittal T2w to save space
  rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT//[\/]/_}_*sag_T2w.* .
else
  rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT//[\/]/_}_*T2w.* .
fi

# Go to subject folder for source images
cd ${SUBJECT}/anat

# ------------------------------------------------------------------------------
# T2w
# ------------------------------------------------------------------------------
# sci-zurich
if [[ $SUBJECT =~ "sub-zh" ]]; then
    # We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
    file_t2="${SUBJECT//[\/]/_}"_acq-sag_T2w
# sci-colorado
else
    file_t2="${SUBJECT}"_T2w
fi
# ------------------------------------------------------------------------------
# T2w
# ------------------------------------------------------------------------------

# Copy GT spinal cord and lesion segmentations
copy_gt_sc "${file_t2}"
copy_gt_lesion "${file_t2}"

# Dilate spinal cord mask
sct_maths -i ${file_t2}_seg.nii.gz -dilate 5 -shape ball -o ${file_sag}_seg-manual_dilate.nii.gz

# Use the dilated mask to crop the original image
sct_crop_image -i ${file_t2}.nii.gz -m ${file_t2}_seg-manual_dilate.nii.gz -o ${file_t2}_crop.nii.gz

# Use the dilated mask to crop the spinal cord GT
sct_crop_image -i ${file_t2}_seg-manual.nii.gz -m ${file_t2}_seg-manual_dilate.nii.gz -o ${file_t2}_seg-manual_crop.nii.gz

# Use the dilated mask to crop the lesion GT
sct_crop_image -i ${file_t2}_lesion-manual.nii.gz -m ${file_t2}_seg-manual_dilate.nii.gz -o ${file_t2}_lesion-manual_crop.nii.gz

# Generate QC to assess the cropping
sct_qc -i ${file_t2}_crop.nii.gz -s ${file_t2}_seg-manual_crop.nii.gz -d ${file_t2}_seg-manual_crop.nii.gz -p sct_deepseg_lesion -plane sagittal -qc ${PATH_QC} -qc-subject ${SUBJECT}
sct_qc -i ${file_t2}_crop.nii.gz -s ${file_t2}_seg-manual_crop.nii.gz -d ${file_t2}_lesion-manual_crop.nii.gz -p sct_deepseg_lesion -plane sagittal -qc ${PATH_QC} -qc-subject ${SUBJECT}

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
