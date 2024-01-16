#!/bin/bash
#
# Run our SCIseg nnUNet model on T2w images the PRAXIS database (site_003 and site_012)
# The script is compatible with all datasets but has to be run separately for each dataset.
# The script:
#   - segments SC using our SCIseg nnUNet model
#   - segments lesion using our SCIseg nnUNet model
#   - copies ground truth (GT) lesion segmentation from derivatives/labels
#   - computes ANIMA segmentation performance metrics between GT and predicted lesion segmentation
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
#  "path_output" : "<PATH_TO_DATASET>_2024-XX-XX",
#  "script"      : "<PATH_TO_REPO>/model_seg_sci/inference_on_other_datasets/run_inference_praxis.sh",
#  "jobs"        : 8,
#  "script_args" : "<PATH_TO_REPO>/model_seg_sci/packaging/run_inference_single_subject.py <PATH_TO_MODEL>/sci-multisite-model"
# }
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

# Segment spinal cord using our nnUNet model
segment_sc_nnUNet(){
  local file="$1"
  local kernel="$2"     # 2d or 3d

  # output file name
  FILESEG="${file}_seg_nnunet_${kernel}"

  # Get the start time
  start_time=$(date +%s)
  # Run SC segmentation
  python ${PATH_NNUNET_SCRIPT} -i ${file}.nii.gz -o ${FILESEG}.nii.gz -path-model ${PATH_NNUNET_MODEL}/nnUNetTrainer__nnUNetPlans__${kernel}_fullres -pred-type sc -use-gpu
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # Generate spinal cord QC report
  sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  # Compute ANIMA segmentation performance metrics
  #compute_anima_metrics ${FILESEG} ${file}_seg-manual.nii.gz
}

# Segment lesion using our nnUNet model
segment_lesion_nnUNet(){
  local file="$1"
  local kernel="$2"     # 2d or 3d
  local plane="$3"      # axial or sagittal plane (for QC)

  # output file name
  FILELESION="${file}_lesion_nnunet_${kernel}"
  # get the sc seg to be used for QC
  FILESEG="${file}_seg_nnunet_3d"

  # Get the start time
  start_time=$(date +%s)
  # Run lesion segmentation
  python ${PATH_NNUNET_SCRIPT} -i ${file}.nii.gz -o ${FILELESION}.nii.gz -path-model ${PATH_NNUNET_MODEL}/nnUNetTrainer__nnUNetPlans__${kernel}_fullres -pred-type lesion -use-gpu
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILELESION},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # Generate lesion QC report
  sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -d ${FILELESION}.nii.gz -p sct_deepseg_lesion -plane ${plane} -qc ${PATH_QC} -qc-subject ${SUBJECT}
  # Compute ANIMA segmentation performance metrics
  compute_anima_metrics ${FILELESION} ${file}_lesion-manual.nii.gz
}


# Copy ground truth (GT) spinal cord or lesion segmentation from derivatives/labels
copy_gt(){
  local file="$1"
  local type="$2"     # lesion or label-SC_mask
  # Construct file name to GT SC or lesion segmentation located under derivatives/labels
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file}_${type}.nii.gz"
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

# Compute ANIMA segmentation performance metrics
compute_anima_metrics(){
  local file_pred="$1"    # segmentation obtained using our nnUNet model
  local file_gt="$2"      # manual GT SC or lesion segmentation
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
# Note: we copy only sagittal T2w image to save space
rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT//[\/]/_}_*acq-sag*_T2w.* .

# Go to subject folder for source images
cd ${SUBJECT}/anat

# ------------------------------------------------------------------------------
# T2w
# ------------------------------------------------------------------------------
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
# If subject is ott004, ott005, sub-hal002 or sub-hal004, add run-01 suffix to the file name
if [[ $SUBJECT =~ "ott004" ]] || [[ $SUBJECT =~ "ott005" ]] || [[ $SUBJECT =~ "sub-hal002" ]] || [[ $SUBJECT =~ "sub-hal004" ]]; then
    file_t2="${SUBJECT//[\/]/_}"_acq-sag_run-01_T2w
# If subject is sub-hal006 or sub-hal026, add run-02 suffix to the file name
elif [[ $SUBJECT =~ "sub-hal006" ]] || [[ $SUBJECT =~ "sub-hal026" ]]; then
    file_t2="${SUBJECT//[\/]/_}"_acq-sag_run-02_T2w
# If subject is ott011 or sub-hal011, add run-03 suffix to the file name
elif [[ $SUBJECT =~ "ott011" ]] || [[ $SUBJECT =~ "sub-hal011" ]]; then
    file_t2="${SUBJECT//[\/]/_}"_acq-sag_run-03_T2w
else
    file_t2="${SUBJECT//[\/]/_}"_acq-sag_T2w
fi

# Check if file_t2 exists
if [[ ! -e ${file_t2}.nii.gz ]]; then
    echo "File ${file_t2}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
    echo "ERROR: File ${file_t2}.nii.gz does not exist. Exiting."
    exit 1
fi

# -----------
# Copy GT
# -----------

# Copy binary GT lesion segmentations from derivatives/labels
copy_gt "${file_t2}" "lesion"

# -----------
# run nnUNet 3D
# -----------

# Segment SC and lesion using our SCIseg nnUNet model
segment_sc_nnUNet "${file_t2}" '3d'
segment_lesion_nnUNet "${file_t2}" '3d' 'sagittal'

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
