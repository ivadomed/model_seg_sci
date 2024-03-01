#!/bin/bash
#
# Generate lesion QC for T2w images.
#
# Dependencies (versions):
# - SCT 6.0 and higher
#
# Usage:
# sct_run_batch -script generate_qc.sh -path-data <PATH-TO-DATASET> -path-output <PATH-TO-OUTPUT> -jobs <num-cpu-cores>

# Lesion and SC labels should be located under:
# PATH_DATA/derivatives/labels/SUBJECT/anat/

# With the following naming convention:
# file_gt="${file}_lesion-manual"
# file_seg="${file}_seg-manual"

# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT


# Retrieve input params and other params
SUBJECT=$1

# get starting time:
start=`date +%s`


# SCRIPT STARTS HERE
# ==============================================================================
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Define variables
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
file="${SUBJECT//[\/]/_}"

# Copy source images
# Note: we use '/./' in order to include the sub-folder 'ses-0X'
rsync -Ravzh $PATH_DATA/./${SUBJECT}/anat/${file}_*T2w.* .

# Copy segmentation ground truths (GT)
mkdir -p derivatives/labels
rsync -Ravzh $PATH_DATA/derivatives/labels/./${SUBJECT}/anat/${file}_*T2w* derivatives/labels/.

# Go to subject folder for source images
cd ${SUBJECT}/anat

# Add suffix corresponding to contrast
# for sci-colorado and sci-paris, use "T2w"
if [[ $PATH_DATA =~ "colorado" ]] || [[ $PATH_DATA =~ "paris" ]]; then
  file=${file}_T2w
# for sci-zurich, use "acq-sag_T2w"
else
  file=${file}_acq-sag_T2w
fi

# Make sure the image metadata is a valid JSON object
if [[ ! -s ${file}.json ]]; then
  echo "{}" >> ${file}.json
fi

# Go to subject folder for segmentation GTs
cd $PATH_DATA_PROCESSED/derivatives/labels/$SUBJECT/anat

# Define variables
file_gt="${file}_lesion-manual"
file_seg="${file}_seg-manual"

# Binarize the GTs because QC only accepts binary images
sct_maths -i ${file_gt}.nii.gz -bin 0 -o ${file_gt}_bin.nii.gz

# Run the QC
sct_qc -i ${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${file}.nii.gz -s ${file_seg}.nii.gz -d ${file_gt}_bin.nii.gz -p sct_deepseg_lesion -plane sagittal -qc ${PATH_QC} -qc-subject ${SUBJECT}

# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
