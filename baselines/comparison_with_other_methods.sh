#!/bin/bash
#
# Compare our nnUNet model with other methods (sct_propseg, sct_deepseg_sc 2d, sct_deepseg_sc 3d) on sci-zurich and
# sci-colorado datasets
#
# Usage:
#     sct_run_batch -c config.json
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

# ------------------------------------------------------------------------------
# CONVENIENCE FUNCTIONS
# ------------------------------------------------------------------------------
# Get ANIMA binaries path
anima_binaries_path=$(grep "^anima = " ~/.anima/config.txt | sed "s/.* = //" | sed 's/\/$//')

# Segment spinal cord and compute ANIMA segmentation performance metrics
segment_sc() {
  local file="$1"
  local contrast="$2"
  local method="$3"     # deepseg or propseg
  local kernel="$4"     # 2d or 3d; only relevant for deepseg

  # Segment spinal cord
  if [[ $method == 'deepseg' ]];then
      FILESEG="${file}_seg_${method}_${kernel}"
      sct_deepseg_sc -i ${file}.nii.gz -o ${FILESEG}.nii.gz -c ${contrast} -kernel ${kernel} -qc ${PATH_QC} -qc-subject ${SUBJECT}
      # Compute ANIMA segmentation performance metrics
      compute_anima_metrics ${FILESEG} ${file}_seg-manual.nii.gz
  elif [[ $method == 'propseg' ]]; then
      FILESEG="${file}_seg_${method}"
      sct_propseg -i ${file}.nii.gz -o ${FILESEG}.nii.gz -c ${contrast} -qc ${PATH_QC} -qc-subject ${SUBJECT}
      # Remove centerline (we don't need it)
      rm ${file}_centerline.nii.gz
      # Compute ANIMA segmentation performance metrics
      compute_anima_metrics ${FILESEG} ${file}_seg-manual.nii.gz
  fi

}

# Compute ANIMA segmentation performance metrics
compute_anima_metrics(){
  # We have to copy qform matrix from seg-manual to the automatically generated segmentation to avoid ITK error:
  # "Description: ITK ERROR: SegmentationMeasuresImageFilter(): Inputs do not occupy the same physical space!"
  # Related to the following issue : https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4135
  sct_image -i ${file}_seg-manual.nii.gz -copy-header ${FILESEG}.nii.gz -o ${FILESEG}_updated_header.nii.gz

  # Compute ANIMA segmentation performance metrics
  # -i : input segmentation
  # -r : GT segmentation
  # -o : output file
  # -d : surface distances evaluation
  # -s : compute metrics to evaluate a segmentation
  # -X : stores results into a xml file.
  ${anima_binaries_path}/animaSegPerfAnalyzer -i ${FILESEG}_updated_header.nii.gz -r ${file}_seg-manual.nii.gz -o ${PATH_RESULTS}/${FILESEG} -d -s -X

  rm ${FILESEG}_updated_header.nii.gz
}

# Copy GT segmentation
copy_gt(){
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

# ------------------------------------------------------------------------------
# SCRIPT STARTS HERE
# ------------------------------------------------------------------------------
# Retrieve input params and other params
SUBJECT=$1

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

# Copy GT segmentation
copy_gt "${file_t2}"

# Check if file_t2 exists
if [[ ! -e ${file_t2}.nii.gz ]]; then
    echo "File ${file_t2}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
    echo "ERROR: File ${file_t2}.nii.gz does not exist. Exiting."
    exit 1
else
    # Segment SC using different methods and compute ANIMA segmentation performance metrics
    segment_sc "${file_t2}" 't2' 'deepseg' '2d'
    segment_sc "${file_t2}" 't2' 'deepseg' '3d'
    segment_sc "${file_t2}" 't2' 'propseg'
fi
