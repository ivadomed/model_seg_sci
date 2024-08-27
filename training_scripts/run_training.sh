#!/bin/bash
#
# Run nnUNetv2_plan_and_preprocess, nnUNetv2_train, and nnUNetv2_predict on the dataset
#
# Example usage:
#     bash run_training.sh <GPU> <dataset_id> <dataset_name> <config> <trainer>
#     bash run_training.sh 1 701 Dataset701_SCIlesions_hemorrhage_RegionBasedSeed42 3d_fullres nnUNetTrainer
#
# Authors: Naga Karthik, Jan Valosek
#

# !!! MODIFY THE FOLLOWING VARIABLES ACCORDING TO YOUR NEEDS !!!
DEVICE=${1}
dataset_id=${2}                        # e.g. 701
dataset_name=${3}                      # e.g. Dataset701_SCIlesions_hemorrhage_RegionBasedSeed42
config=${4}                            # e.g. 3d_fullres or 2d
nnunet_trainer=${5}                    # default: nnUNetTrainer
                                       # other options: nnUNetTrainer_250epochs, nnUNetTrainer_2000epochs,
                                       # nnUNetTrainerDA5, nnUNetTrainerDA5_DiceCELoss_noSmooth

# Check whether config is valid, if not, exit
if [[ ${config} != "2d" && ${config} != "3d_fullres" ]]; then
    echo "Invalid configuration. Please use either 2d or 3d_fullres."
    exit 1
fi

# Check whether nnunet_trainer is valid, if not, exit
available_trainers=("nnUNetTrainer" "nnUNetTrainer_250epochs" "nnUNetTrainer_2000epochs" "nnUNetTrainerDA5" "nnUNetTrainerDA5_DiceCELoss_noSmooth")
if [[ ! " ${available_trainers[@]} " =~ " ${nnunet_trainer} " ]]; then
    echo "Invalid nnUNet trainer. Please use one of the following: ${available_trainers[@]}"
    exit 1
fi

# Select number of folds here
# folds=(0 1 2 3 4)
# folds=(0 1 2)
folds=(0)

echo "-------------------------------------------------------"
echo "Running preprocessing and verifying dataset integrity"
echo "-------------------------------------------------------"

nnUNetv2_plan_and_preprocess -d ${dataset_id} --verify_dataset_integrity -c ${config}

for fold in ${folds[@]}; do
    echo "-------------------------------------------"
    echo "Training on Fold $fold"
    echo "-------------------------------------------"

    # training
    CUDA_VISIBLE_DEVICES=${DEVICE} nnUNetv2_train ${dataset_id} ${config} ${fold} -tr ${nnunet_trainer}

    echo ""
    echo "-------------------------------------------"
    echo "Training completed, Testing on Fold $fold"
    echo "-------------------------------------------"

#    # inference
#    CUDA_VISIBLE_DEVICES=${DEVICE} nnUNetv2_predict -i ${nnUNet_raw}/${dataset_name}/imagesTs -tr ${nnunet_trainer} -o ${nnUNet_results}/${nnunet_trainer}__nnUNetPlans__${config}/fold_${fold}/test -d ${dataset_id} -f ${fold} -c ${config}
#
#    echo ""
#    echo "-------------------------------------------"
#    echo " Inference completed on Fold $fold"
#    echo "-------------------------------------------"

done