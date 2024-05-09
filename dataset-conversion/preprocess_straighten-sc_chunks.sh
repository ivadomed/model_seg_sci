#!/bin/bash
#
# Preprocess data.
#
# Dependencies (versions):
# - SCT (6.3.0)
#
# Usage:
# sct_run_batch -script preprocess_data.sh -path-data <PATH-TO-DATASET> -path-output <PATH-TO-OUTPUT> -jobs <num-cpu-cores>

# Manual segmentations or labels should be located under:
# PATH_DATA/derivatives/labels/SUBJECT/ses-0X/anat/

# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"

# Global variables
CENTERLINE_METHOD="svm"  # method sct_deepseg_sc uses for centerline extraction: 'svm', 'cnn'

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT


# CONVENIENCE FUNCTIONS
# ======================================================================================================================

segment_if_does_not_exist() {
  ###
  #  This function checks if a manual spinal cord segmentation file already exists, then:
  #    - If it does, copy it locally.
  #    - If it doesn't, perform automatic spinal cord segmentation.
  #  This allows you to add manual segmentations on a subject-by-subject basis without disrupting the pipeline.
  ###
  local file="$1"
  local contrast="$2"
  local centerline_method="$3"
  # Update global variable with segmentation file name
  FILESEG="${file}_seg"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILESEG}-manual.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic segmentation."
    # Segment spinal cord based on the specified centerline method
    if [[ $centerline_method == "cnn" ]]; then
      sct_deepseg_sc -i ${file}.nii.gz -c $contrast -brain 1 -centerline cnn -qc ${PATH_QC} -qc-subject ${SUBJECT}
    elif [[ $centerline_method == "svm" ]]; then
      sct_deepseg_sc -i ${file}.nii.gz -c $contrast -centerline svm -qc ${PATH_QC} -qc-subject ${SUBJECT}
    else
      echo "Centerline extraction method = ${centerline_method} is not recognized!"
      exit 1
    fi
  fi
}

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

# Copy BIDS-required files to processed data folder (e.g. list of participants)
if [[ ! -f "participants.tsv" ]]; then
  rsync -avzh $PATH_DATA/participants.tsv .
fi
if [[ ! -f "participants.json" ]]; then
  rsync -avzh $PATH_DATA/participants.json .
fi
if [[ ! -f "dataset_description.json" ]]; then
  rsync -avzh $PATH_DATA/dataset_description.json .
fi
if [[ ! -f "README" ]]; then
  rsync -avzh $PATH_DATA/README .
fi

# Copy source images
# Note: we use '/./' in order to include the sub-folder 'ses-0X'
rsync -Ravzh $PATH_DATA/./$SUBJECT .

# Copy segmentation ground truths (GT)
# mkdir -p derivatives/labels
rsync -Ravzh $PATH_DATA/derivatives/labels/./$SUBJECT . # derivatives/labels/.

# print the current contents of the PATH_DATA_PROCESSED folder
echo "The contents of the PATH_DATA_PROCESSED folder are:"
ls -l  

# Define variables
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
# file="${SUBJECT//[\/]/_}"

# Copy BIDS-required files to processed data folder
#rsync -avzh $PATH_DATA/{participants.tsv,participants.json,dataset_description.json,README} .

# Loop over all axial files in the subject's anatomy directory
# cd ${SUBJECT}/anat
cd ${SUBJECT}/anat
shopt -s nullglob # Handle the case where no files match the pattern
counter=1  # Initialize counter at 1

for file in $(find "$PWD" -type f -name "*acq-ax*" ! -name "*seg*" ! -name "*lesion*" -name "*.nii.gz" | sort)
do
    echo "Processing $file"
    base_name=$(echo "$file" | sed 's/\.nii\.gz$//')

    # Make sure the image metadata is a valid JSON object
    if [[ ! -s ${base_name}.json ]]; then
      echo "{}" > ${base_name}.json
    fi

    # Process with incrementing warp file names
    sct_straighten_spinalcord -i ${file} -s ${base_name}_seg-manual.nii.gz -o ${base_name}_desc-straightened.nii.gz
    mv warp_curve2straight.nii.gz warp_curve2straight_chunk-${counter}.nii.gz
    mv warp_straight2curve.nii.gz warp_straight2curve_chunk-${counter}.nii.gz
    mv straight_ref.nii.gz straight_ref_chunk-${counter}.nii.gz
    sct_apply_transfo -i ${base_name}_seg-manual.nii.gz -d ${base_name}_desc-straightened.nii.gz -w warp_curve2straight_chunk-${counter}.nii.gz -o ${base_name}_seg-manual_desc-straightened.nii.gz
    sct_apply_transfo -i ${base_name}_lesion-manual.nii.gz -d ${base_name}_desc-straightened.nii.gz -w warp_curve2straight_chunk-${counter}.nii.gz -o ${base_name}_lesion-manual_desc-straightened.nii.gz

    # Threshold and other post-processing as needed
    sct_maths -i ${base_name}_seg-manual_desc-straightened.nii.gz -o ${base_name}_seg-manual_desc-straightened.nii.gz -thr 0.10
    sct_maths -i ${base_name}_lesion-manual_desc-straightened.nii.gz -o ${base_name}_lesion-manual_desc-straightened.nii.gz -thr 0.10

    # Increment the counter after processing each file
    ((counter++))
done

shopt -u nullglob # Turn off nullglob to return to normal glob behavior

# Go back to the root output path
cd $PATH_OUTPUT

# Create and populate clean data processed folder for training
PATH_DATA_PROCESSED_CLEAN="${PATH_DATA_PROCESSED}_clean"

# Copy over required BIDs files
mkdir -p $PATH_DATA_PROCESSED_CLEAN $PATH_DATA_PROCESSED_CLEAN/${SUBJECT} $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/anat
rsync -avzh $PATH_DATA_PROCESSED/dataset_description.json $PATH_DATA_PROCESSED_CLEAN/
rsync -avzh $PATH_DATA_PROCESSED/participants.* $PATH_DATA_PROCESSED_CLEAN/
rsync -avzh $PATH_DATA_PROCESSED/README $PATH_DATA_PROCESSED_CLEAN/

# Images
for file in $PATH_DATA_PROCESSED/${SUBJECT}/anat/ -name "*chunk*straight*T2w*" ! -name "*manu*" 
do
# Image
  mkdir -p $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/anat/
  file_clean="${file/$PATH_DATA_PROCESSED/$PATH_DATA_PROCESSED_CLEAN}"
  rsync -avzh $file $file_clean
done

# Labels
for file in $PATH_DATA_PROCESSED/${SUBJECT}/anat/ -name "*chunk*manual*straight*"
do
# Labels
  mkdir -p $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/anat/
  file_clean="${file/$PATH_DATA_PROCESSED/$PATH_DATA_PROCESSED_CLEAN/derivatives/labels}"
  rsync -avzh $file $file_clean
done

# Warp
for file in $PATH_DATA_PROCESSED/${SUBJECT}/anat/ -name "*warp*"
do
# Warp
  mkdir -p $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/anat/
  file_clean="${file/$PATH_DATA_PROCESSED/$PATH_DATA_PROCESSED_CLEAN/derivatives/labels}"
  rsync -avzh $file $file_clean
done

# Ref
for file in $PATH_DATA_PROCESSED/${SUBJECT}/anat/ -name "*ref*"
do
# Ref
  mkdir -p $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/anat/
  file_clean="${file/$PATH_DATA_PROCESSED/$PATH_DATA_PROCESSED_CLEAN/derivatives/labels}"
  rsync -avzh $file $file_clean
done

# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"