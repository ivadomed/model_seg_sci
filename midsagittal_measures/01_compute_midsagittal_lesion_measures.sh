#!/bin/bash
#
# Compute midsagittal lesion measures (length, width, and tissue bridges) from manual and automated segmentations.
#
# The script does the following:
#   1. Copies manual segmentations of spinal cord and lesions from derivatives/labels
#   2. Computes midsagittal lesion measures based on the manual segment
#   3. Segments spinal cord and lesions using SCIsegV2 model
#   4. Computes midsagittal lesion measures based on the SCIsegV2
#
# NOTE: This script requires SCT v7.0 or higher (due to the new sct_deepseg syntax).

# NOTE: The script is meant to be run on GPU (see `CUDA_VISIBLE_DEVICES=1 SCT_USE_GPU=1 sct_deepseg ...` below).
#
# Usage:
#     sct_run_batch -config config-01_compute_midsagittal_lesion_length_and_width.json
#
# Example of config-01_compute_midsagittal_lesion_length_and_width.json:
# {
#  "path_data"   : "<PATH_TO_DATASET>",
#  "path_output" : "<PATH_TO_DATASET>_2024-09-20",
#  "script"      : "<PATH_TO_REPO>/model_seg_sci/midsagittal_measures/01_compute_midsagittal_lesion_measures.sh",
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
# Author: Jan Valosek
#

# Uncomment for full verbose
set -x

# Immediately exit if error
#set -e -o pipefail

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
# Manual GT
# ------------------------------------
# Copy GT SC and lesion segmentations from derivatives/labels
copy_gt "${file_t2}" "seg"
copy_gt "${file_t2}" "lesion"

# Binarize GT lesion segmentation (sct_analyze_lesion requires binary mask until https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4120 is fixed)
sct_maths -i ${file_t2}_lesion-manual.nii.gz -bin 0 -o ${file_t2}_lesion-manual_bin.nii.gz

# Generate sagittal lesion QC report
sct_qc -i ${file_t2}.nii.gz -d ${file_t2}_lesion-manual_bin.nii.gz -s ${file_t2}_seg-manual.nii.gz -p sct_deepseg_lesion -plane sagittal -qc ${PATH_QC} -qc-subject "lesion_manual"

# Compute the midsagittal lesion length and width based on the spinal cord and lesion segmentations obtained manually
status=0
sct_analyze_lesion -m ${file_t2}_lesion-manual_bin.nii.gz -s ${file_t2}_seg-manual.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT} || status=$?
# If status is not zero, sct_analyze_lesion failed (e.g., because there is no lesion in the GT segmentation)
if [ $status -ne 0 ]; then
    echo "❌No lesion found in manual GT segmentation for ${file_t2}" >> ${PATH_LOG}/manual_GT_analysis.log
    exit 0
else
  # If sct_analyze_lesion finished successfully, the outputs are:
  #   - ${file_t2}_lesion-manual_bin_label.nii.gz: 3D mask of the segmented lesion with lesion IDs (1, 2, 3, etc.)
  #   - ${file_t2}_lesion-manual_bin_analysis.xlsx: XLSX file containing the morphometric measures
  #   - ${file_t2}_lesion-manual_bin_analysis.pkl: Python Pickle file containing the morphometric measures
  # Remove pickle file -- we only need the XLSX file
  rm ${file_t2}_lesion-manual_bin_analysis.pkl

  # Copy the XLSX file to the results folder
  cp ${file_t2}_lesion-manual_bin_analysis.xlsx ${PATH_RESULTS}
  echo "✅ ${file_t2}_lesion-manual_bin_analysis.xlsx created" >> ${PATH_LOG}/manual_GT_analysis.log
fi
# ----------------------------
# SCIsegV2
# ----------------------------
# Segment the spinal cord and lesions using SCIsegV2
#CUDA_VISIBLE_DEVICES=1 SCT_USE_GPU=1 sct_deepseg lesion_sci_t2 -i ${file_t2}.nii.gz -largest 1 -qc ${PATH_QC} -qc-subject ${SUBJECT}
sct_deepseg lesion_sci_t2 -i ${file_t2}.nii.gz -largest 1 -qc ${PATH_QC} -qc-subject ${SUBJECT}
# The outputs are:
#   - ${file_t2}_sc_seg.nii.gz:  3D binary mask of the segmented spinal cord
#   - ${file_t2}_lesion_seg.nii.gz: 3D binary mask of the segmented lesion
# Rename the SC seg to make clear it comes from the SCIsegV2 model
mv ${file_t2}_sc_seg.nii.gz ${file_t2}_sc_seg_SCIsegV2.nii.gz

# Generate sagittal lesion QC report (because sct_deepseg produces only axial QC report showing both SC and lesion).
# But we want to show only the lesion segmentation in the QC report on sagittal slices.
sct_qc -i ${file_t2}.nii.gz -d ${file_t2}_lesion_seg.nii.gz -s ${file_t2}_sc_seg_SCIsegV2.nii.gz -p sct_deepseg_lesion -plane sagittal -qc ${PATH_QC} -qc-subject "lesion_SCIsegV2"

# Compute the midsagittal lesion length and width based on the spinal cord and lesion segmentations obtained using SCIsegV2
status=0
sct_analyze_lesion -m ${file_t2}_lesion_seg.nii.gz -s ${file_t2}_sc_seg_SCIsegV2.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT} || status=$?
# If status is not zero, sct_analyze_lesion failed (e.g., because there is no lesion in the GT segmentation)
if [ $status -ne 0 ]; then
    echo "❌No lesion segmented by SCIsegV2 for ${file_t2}" >> ${PATH_LOG}/SCIsegV2_predictions_analysis.log
    exit 0
else
  # If sct_analyze_lesion finished successfully, the outputs are:
  #   - ${file_t2}_lesion_seg_label.nii.gz: 3D mask of the segmented lesion with lesion IDs (1, 2, 3, etc.)
  #   - ${file_t2}_lesion_seg_analysis.xlsx: XLSX file containing the morphometric measures
  #   - ${file_t2}_lesion_seg_analysis.pkl: Python Pickle file containing the morphometric measures
  # Remove pickle file -- we only need the XLSX file
  rm ${file_t2}_lesion_seg_analysis.pkl

  # Rename the files to make clear they come from the SCIsegV2 model
  mv ${file_t2}_lesion_seg_label.nii.gz ${file_t2}_lesion_seg_label_SCIsegV2.nii.gz
  mv ${file_t2}_lesion_seg_analysis.xlsx ${file_t2}_lesion_seg_analysis_SCIsegV2.xlsx
  # Copy the XLSX file to the results folder
  cp ${file_t2}_lesion_seg_analysis_SCIsegV2.xlsx ${PATH_RESULTS}
  echo "✅ ${file_t2}_lesion_seg_analysis_SCIsegV2.xlsx created" >> ${PATH_LOG}/SCIsegV2_predictions_analysis.log
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
