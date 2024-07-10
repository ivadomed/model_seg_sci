#!/bin/bash
#
# Compute tissue bridges using sct_analyze_lesion from lesion and spinal cord segmentations obtained using SCIsegV2
#
# NOTE: This script requires the following SCT branch:
# https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4489
#
# Usage:
#     ./<PATH_TO_REPO>/model_seg_sci/tissue_bridges/01a_compute_tissue_bridges_from_GT.sh <PATH_in>
#
# NOTE: I do not use sct_run_batch here because SCIsegV2 outputs are not in BIDS format
#
# Author: Jan Valosek
#

# Uncomment for full verbose
# set -x

# Immediately exit if error
set -e -o pipefail

# Check whether the user has provided input arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 <PATH_IN>"
    exit 1
fi

# Check if PATH_IN exists, if so, go to that directory
if [ -d $1 ]; then
    cd $1
else
    echo "Directory $1 does not exist. Exiting."
    exit 1
fi

# Check if PATH_OUT exists, if not, create it
PATH_RESULTS=${1}/results
if [ ! -d $PATH_RESULTS ]; then
    mkdir -p $PATH_RESULTS
fi

# ------------------------------------------------------------------------------
# T2w
# ------------------------------------------------------------------------------
# Loop across all T2w nii images in the directory
for file_t2w in *.nii.gz;do

    echo "Processing: ${file_t2w}"

    # Strip the suffix
    file_t2w=${file_t2w/.nii.gz/}

    # SCIsegV2 output contains voxels with values 1 for SC and voxels with values 2 for lesion --> separate them
    # into two files
    sct_maths -i ${file_t2w}.nii.gz -bin 0 -o ${file_t2w}_seg.nii.gz
    sct_maths -i ${file_t2w}.nii.gz -bin 1 -o ${file_t2w}_lesion.nii.gz

    # Analyze lesion segmentation
    sct_analyze_lesion -m ${file_t2w}_lesion.nii.gz -s ${file_t2w}_seg.nii.gz -ofolder ${PATH_RESULTS}

done

# ------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------
