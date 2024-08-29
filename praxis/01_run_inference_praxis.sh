#!/bin/bash
#
# Run the SCIsegV2 model (part of SCT v6.4) on T2w images from the PRAXIS database (e.g., site_003, site_012, ...)
# The script is compatible with all PRAXIS datasets but has to be run separately for each dataset.
#
# NOTE: since most of the subjects have multiple runs of T2w images, we need to hardcode the run number for each
# subject into the "T2w" section of this script. I am aware that this is not the best solution..., see the comment
# above the if-else block below.
#
# The script:
#   - segments spinal cord and intramedullary lesions the SCIsegV2 nnUNet model and generates QC
#     Note: a single axial QC report contains both SC and lesion segmentations
#
# Usage:
#     sct_run_batch -config config.json
#
# Example of config.json:
# {
#  "path_data"   : "<PATH_TO_DATASET>",
#  "path_output" : "<PATH_TO_DATASET>_2024-XX-XX",
#  "script"      : "<PATH_TO_REPO>/model_seg_sci/praxis/01_run_inference_praxis.sh",
#  "jobs"        : 8
# }
#
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

echo "SUBJECT: ${SUBJECT}"

# ------------------------------------------------------------------------------
# CONVENIENCE FUNCTIONS
# ------------------------------------------------------------------------------

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
# Now, we use T2w images from different runs for different subjects
# This depends on a lot of factors (site, artifact, etc.); see an issue describing suitable images for each site at spineimage.ca/
# The if-else block below is suboptimal and needs to be updated with each new site --> find a better solution
# run-01
if [[ $SUBJECT =~ "ott004" ]] || [[ $SUBJECT =~ "ott005" ]] || [[ $SUBJECT =~ "hal002" ]] || [[ $SUBJECT =~ "hal004" ]]  || [[ $SUBJECT =~ "ham" ]] || [[ $SUBJECT =~ "que002" ]] || [[ $SUBJECT =~ "que008" ]]; then
    file_t2="${SUBJECT//[\/]/_}"_acq-sag_run-01_T2w
# run-02
elif [[ $SUBJECT =~ "sub-hal006" ]] || [[ $SUBJECT =~ "hal026" ]] || [[ $SUBJECT =~ "que012" ]]; then
    file_t2="${SUBJECT//[\/]/_}"_acq-sag_run-02_T2w
# run-03
elif [[ $SUBJECT =~ "ott011" ]] || [[ $SUBJECT =~ "hal011" ]]; then
    file_t2="${SUBJECT//[\/]/_}"_acq-sag_run-03_T2w
# run-04
elif [[ $SUBJECT =~ "que004" ]]; then
    file_t2="${SUBJECT//[\/]/_}"_acq-sag_run-04_T2w
# "no run"
else
    file_t2="${SUBJECT//[\/]/_}"_acq-sag_T2w
fi

# For site-007, we have >100 subjects --> it is not feasible to specify the run for each subject. We thus check whether
# GT lesion mask exists under derivatives/labels and if so, we determine the run number from the GT lesion mask name
if [[ $PATH_DATA =~ "site-007" ]]; then
    # Check if GT lesion mask exists
    FILELESIONMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file_t2}_lesion.nii.gz"
    if [[ -e $FILELESIONMANUAL ]]; then
        # Extract run number from the GT lesion mask name
        run_number=$(echo $FILELESIONMANUAL | grep -oP 'run-\d{2}' | cut -d'-' -f2)
        # Check if run_number is not empty string
        # (because there might be no run number in the GT lesion mask name, if such a case, use file_t2 as is)
        if [[ -n $run_number ]]; then
            file_t2="${SUBJECT//[\/]/_}"_run-${run_number}
        fi
    fi
fi

# Moreover, for que, use acq-sagittal instead of acq-sag
if [[ $SUBJECT =~ "que" ]]; then
    file_t2="${file_t2//acq-sag/acq-sagittal}"
fi

# Check if file_t2 exists
if [[ ! -e ${file_t2}.nii.gz ]]; then
    echo "File ${file_t2}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
    echo "ERROR: File ${file_t2}.nii.gz does not exist. Exiting."
    exit 1
fi

# ---------------
# run SCIsegV2
# ---------------

# Segment SC and lesion using SCIsegV2 (part of SCT v6.4)
# Note: a single axial QC report contains both SC and lesion segmentations
# Note: we use CUDA_VISIBLE_DEVICES=0 SCT_USE_GPU=1 to run the inference on GPU 0; details:
# https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4421#issuecomment-2263344151
CUDA_VISIBLE_DEVICES=0 SCT_USE_GPU=1 sct_deepseg -i ${file_t2}.nii.gz -task seg_sc_lesion_t2w_sci -qc ${PATH_QC} -qc-subject ${SUBJECT}
# Note: outputs are ${file_t2}_sc_seg.nii.gz and ${file_t2}_lesion_seg.nii.gz
# Generate sagittal lesion QC report
sct_qc -i ${file_t2}.nii.gz -s ${file_t2}_sc_seg.nii.gz -d ${file_t2}_lesion_seg.nii.gz -p sct_deepseg_lesion -plane sagittal -qc ${PATH_QC} -qc-subject ${SUBJECT}

# ---------------
# Copy GT manual lesion mask and generate QC report (to compare it with the model output)
# ---------------

# Copy binary GT lesion segmentations from derivatives/labels
copy_gt "${file_t2}" "lesion"
# Generate sagittal lesion QC report
# Note: we need to specify also SC segmentation (`-s`) to crop the image properly, so we're using the SC segmentation from the model output
sct_qc -i ${file_t2}.nii.gz -s ${file_t2}_sc_seg.nii.gz -d ${file_t2}_lesion-manual.nii.gz -p sct_deepseg_lesion -plane sagittal -qc ${PATH_QC} -qc-subject ${SUBJECT}

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
