#!/bin/bash
#
# Run inference using our SCIseg nnUNet model on dcm-zurich-lesions T2w axial images
# The script:
#     - segments spinal cord using our 3D nnUNet model and generates QC report
#     - segments lesions using our 3D nnUNet model and generates QC report
#     - copies manual GT lesion segmentation and generates QC report
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
#  "path_output" : "<PATH_TO_DATASET>_2023-08-22",
#  "script"      : "<PATH_TO_REPO>/model_seg_sci/baselines/run_inference_sci-zurich_axial.sh",
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
# Get ANIMA binaries path
anima_binaries_path=$(grep "^anima = " ~/.anima/config.txt | sed "s/.* = //" | sed 's/\/$//')

# Compute ANIMA segmentation performance metrics
compute_anima_metrics(){
  local file_pred="$1"
  local file_gt="$2"

  # We have to copy qform matrix from seg-manual to the automatically generated segmentation to avoid ITK error:
  # "Description: ITK ERROR: SegmentationMeasuresImageFilter(): Inputs do not occupy the same physical space!"
  # Related to the following issue : https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4135
  sct_image -i ${file_gt}.nii.gz -copy-header ${file_pred}.nii.gz -o ${file_pred}_updated_header.nii.gz

  # Compute ANIMA segmentation performance metrics
  # -i : input segmentation
  # -r : GT segmentation
  # -o : output file
  # -d : surface distances evaluation
  # -s : compute metrics to evaluate a segmentation
  # -l : lesion detection evaluation
  # -X : stores results into a xml file.
  ${anima_binaries_path}/animaSegPerfAnalyzer -i ${file_pred}_updated_header.nii.gz -r ${file_gt}.nii.gz -o ${PATH_RESULTS}/${file_pred} -d -s -l -X

  rm ${file_pred}_updated_header.nii.gz
}


# Segment spinal cord using our nnUNet model
segment_sc_nnUNet(){
  local file="$1"
  local kernel="$2"     # 2d or 3d

  FILESEG="${file}_sc_nnunet_${kernel}"

  # Get the start time
  start_time=$(date +%s)
  # Run SC segmentation
  python ${PATH_NNUNET_SCRIPT} -i ${file}.nii.gz -o ${FILESEG}.nii.gz -path-model ${PATH_NNUNET_MODEL}/nnUNetTrainerDiceCELoss_noSmooth__nnUNetPlans__${kernel} -pred-type sc -use-gpu
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # Generate QC report
  sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

  # Compute ANIMA segmentation performance metrics
  compute_anima_metrics ${FILESEG} ${file}_label-SC_mask-manual

}

# Segment lesion using our nnUNet model
segment_lesion_nnUNet(){
  local file="$1"
  local kernel="$2"     # 2d or 3d
  local file_seg="$3"   # SC segmentation (used for QC)

  FILESEGLESION="${file}_lesion_nnunet_${kernel}"

  # Get the start time
  start_time=$(date +%s)
  # Run SC segmentation
  python ${PATH_NNUNET_SCRIPT} -i ${file}.nii.gz -o ${FILESEGLESION}.nii.gz -path-model ${PATH_NNUNET_MODEL}/nnUNetTrainerDiceCELoss_noSmooth__nnUNetPlans__${kernel} -pred-type lesion -use-gpu
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILESEGLESION},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # Generate QC report
  sct_qc -i ${file}.nii.gz -s ${file_seg}.nii.gz -d ${FILESEGLESION}.nii.gz -p sct_deepseg_lesion -plane axial -qc ${PATH_QC} -qc-subject ${SUBJECT}

  # Compute ANIMA segmentation performance metrics
  compute_anima_metrics ${FILESEGLESION} ${file}_label-lesion
}

# Copy GT lesion segmentation (located under derivatives/labels)
copy_gt(){
  local file="$1"
  # Construct file name to GT segmentation located under derivatives/labels
  FILELESIONMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file}_label-lesion.nii.gz"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file}_label-SC_mask-manual.nii.gz"
  echo ""
  echo "Looking for manual segmentation: $FILELESIONMANUAL"
  if [[ -e $FILELESIONMANUAL ]]; then
      echo "Found! Copying ..."
      rsync -avzh $FILELESIONMANUAL ${file}_label-lesion.nii.gz
      rsync -avzh $FILESEGMANUAL ${file}_label-SC_mask-manual.nii.gz
  else
      echo "File ${FILELESIONMANUAL} does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: Manual GT segmentation ${FILELESIONMANUAL} does not exist. Exiting."
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
# copy only axial T2w to save space
rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT//[\/]/_}_*ax_T2w.* .

# Go to subject folder for source images
cd ${SUBJECT}/anat

# ------------------------------------------------------------------------------
# T2w axial
# ------------------------------------------------------------------------------

# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
file_t2="${SUBJECT//[\/]/_}"_acq-ax_T2w

# Check if file_t2 exists
if [[ ! -e ${file_t2}.nii.gz ]]; then
    echo "File ${file_t2}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
    echo "ERROR: File ${file_t2}.nii.gz does not exist. Exiting."
    exit 1
fi

# Copy GT lesion segmentation (to generate GT lesion QC report)
copy_gt "${file_t2}"

# Segment spinal cord using our nnUNet SCIseg model
CUDA_VISIBLE_DEVICES=2 segment_sc_nnUNet "${file_t2}" '3d_fullres'

# Segment lesion using our nnUNet SCIlesion model
# Note: SC seg is passed to generate QC report
CUDA_VISIBLE_DEVICES=2 segment_lesion_nnUNet "${file_t2}" '3d_fullres' "${file_t2}_sc_nnunet_3d_fullres"

# Generate QC report for manual lesion segmentation
# Note: there is no manual SC segmentation, so we use the nnUNet SC segmentation
sct_qc -i ${file_t2}.nii.gz -s ${file_t2}_sc_nnunet_3d_fullres.nii.gz -d ${file_t2}_label-lesion.nii.gz -p sct_deepseg_lesion -plane axial -qc ${PATH_QC} -qc-subject ${SUBJECT} -qc-dataset ${SUBJECT}

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