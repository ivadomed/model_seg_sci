#!/bin/bash
#
# Run sct_analyze_lesion on:
#     - GT lesion and spinal cord segmentations (located under derivatives/labels)
#     - predicted lesion and spinal cord segmentations (using our nnUNet 3D model)
#
# Note: subjects from both datasets have to be located in the same BIDS-like folder, example:
# ├── derivatives
# │	 └── labels
# │	     ├── sub-5416   # sci-colorado subject
# │	     │	 └── anat
# │	     │	     ├── sub-5416_T2w_lesion-manual.json
# │	     │	     ├── sub-5416_T2w_lesion-manual.nii.gz
# │	     │	     ├── sub-5416_T2w_seg-manual.json
# │	     │	     └── sub-5416_T2w_seg-manual.nii.gz
# │	     └── sub-zh01   # sci-zurich subject
# │	         └── ses-01
# │	             └── anat
# │	                 ├── sub-zh01_ses-01_acq-sag_T2w_lesion-manual.json
# │	                 ├── sub-zh01_ses-01_acq-sag_T2w_lesion-manual.nii.gz
# │	                 ├── sub-zh01_ses-01_acq-sag_T2w_seg-manual.json
# │	                 └── sub-zh01_ses-01_acq-sag_T2w_seg-manual.nii.gz
# ├── sub-5416    # sci-colorado subject
# │	 └── anat
# │	     ├── sub-5416_T2w.json
# │	     └── sub-5416_T2w.nii.gz
# └── sub-zh01    # sci-zurich subject
#    └── ses-01
#        └── anat
#            ├── sub-zh01_ses-01_acq-sag_T2w.json
#            └── sub-zh01_ses-01_acq-sag_T2w.nii.gz
#
# Note: conda environment with nnUNetV2 is required to run this script.
# For details how to install nnUNetV2, see:
# https://github.com/ivadomed/utilities/blob/main/quick_start_guides/nnU-Net_quick_start_guide.md#installation
#
# Usage:
#     sct_run_batch -config config.json
#
# Example of config.json:
# {
#  "path_data"   : "<PATH_TO_DATASET>",
#  "path_output" : "<PATH_TO_DATASET>_2023-08-18",
#  "script"      : "<PATH_TO_REPO>/model_seg_sci/baselines/01_analyze_lesions.sh",
#  "jobs"        : 8,
#  "script_args" : "<PATH_TO_REPO>/model_seg_sci/packaging/run_inference_single_subject.py <PATH_TO_MODEL>/sci-multisite-model"
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
PATH_NNUNET_SCRIPT=$2
PATH_NNUNET_MODEL=$3

echo "SUBJECT: ${SUBJECT}"
echo "PATH_NNUNET_SCRIPT: ${PATH_NNUNET_SCRIPT}"
echo "PATH_NNUNET_MODEL: ${PATH_NNUNET_MODEL}"

# ------------------------------------------------------------------------------
# CONVENIENCE FUNCTIONS
# ------------------------------------------------------------------------------

# Segment spinal cord using our nnUNet model
segment_sc_nnUNet(){
  local file="$1"
  local kernel="$2"     # 2d or 3d

  FILESEG="${file}_seg_nnunet_${kernel}"

  # Get the start time
  start_time=$(date +%s)
  # Run SC segmentation
  python ${PATH_NNUNET_SCRIPT} -i ${file}.nii.gz -o ${FILESEG}.nii.gz -path-model ${PATH_NNUNET_MODEL}/nnUNetTrainerDiceCELoss_noSmooth__nnUNetPlans__${kernel}_fullres -pred-type sc
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # Generate QC report
  sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
}

# Segment lesion using our nnUNet model
segment_lesion_nnUNet(){
  local file="$1"
  local kernel="$2"     # 2d or 3d

  FILELESION="${file}_lesion_nnunet_${kernel}"

  # get the GT sc seg to be used for QC
  FILEGTSEG="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file}_seg-manual"

  # Get the start time
  start_time=$(date +%s)
  # Run lesion segmentation
  python ${PATH_NNUNET_SCRIPT} -i ${file}.nii.gz -o ${FILELESION}.nii.gz -path-model ${PATH_NNUNET_MODEL}/nnUNetTrainerDiceCELoss_noSmooth__nnUNetPlans__${kernel}_fullres -pred-type lesion
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILELESION},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  if [[ $SUBJECT =~ "sub-zh" ]]; then
    # Generate sagittal QC report
    sct_qc -i ${file}.nii.gz -s ${FILEGTSEG}.nii.gz -d ${FILELESION}.nii.gz -p sct_deepseg_lesion -plane sagittal -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    # Generate axial QC report
    sct_qc -i ${file}.nii.gz -s ${FILEGTSEG}.nii.gz -d ${FILELESION}.nii.gz -p sct_deepseg_lesion -plane axial -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi
}


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

# ------------------------------------
# nnUNet 3D
# ------------------------------------

# Segment SC and lesion using our nnUNet model
segment_sc_nnUNet "${file_t2}" '3d'
segment_lesion_nnUNet "${file_t2}" '3d'

# Analyze SCI lesion segmentation obtained using our nnUNet model
sct_analyze_lesion -m ${file_t2}_lesion_nnunet_3d.nii.gz -s ${file_t2}_seg_nnunet_3d.nii.gz -ofolder ${PATH_RESULTS}

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
