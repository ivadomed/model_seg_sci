#!/bin/bash
# Training nnUNet on just the Zurich dataset

# define arguments for nnUNet
dataset_name="Dataset276_tSCICombinedSeed7"
dataset_num="276"
nnunet_trainer="nnUNetTrainer_2000epochs"       # default: nnUNetTrainer
configuration="3d_fullres"                      # for 2D training, use "2d"
cuda_visible_devices=1                      
# folds=(1 2)
folds=(3)


# echo "-------------------------------------------------------"
# echo "Running preprocessing and verifying dataset integrity"
# echo "-------------------------------------------------------"
# nnUNetv2_plan_and_preprocess -d 511 --verify_dataset_integrity -c 3d_fullres


for fold in ${folds[@]}; do
    echo "-------------------------------------------"
    echo "Training on Fold $fold"
    echo "-------------------------------------------"
    
    # training
    CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_train ${dataset_num} $configuration $fold -tr ${nnunet_trainer}
    
    echo ""
    echo "-------------------------------------------"
    echo "Training completed, Testing on Fold $fold"
    echo "-------------------------------------------"

    # run inference on zurich test set
    CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_predict -i ${nnUNet_raw}/${dataset_name}/imagesTsZur -tr ${nnunet_trainer} -o ~/nnunet-v2/nnUNet_results/${dataset_name}/${nnunet_trainer}__nnUNetPlans__${configuration}/fold_${fold}/testZur -d ${dataset_num} -f $fold -c ${configuration} # -step_size 0.9 --disable_tta

    # run inference on colorado test set
    CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_predict -i ${nnUNet_raw}/${dataset_name}/imagesTsCol -tr ${nnunet_trainer} -o ~/nnunet-v2/nnUNet_results/${dataset_name}/${nnunet_trainer}__nnUNetPlans__${configuration}/fold_${fold}/testCol_v2 -d ${dataset_num} -f $fold -c ${configuration} # -step_size 0.9 --disable_tta
    
    echo "-------------------------------------------"
    echo "Running ANIMA evaluation on Zurich Test set"
    echo "-------------------------------------------"

    python testing/compute_anima_metrics.py --pred-folder ~/nnunet-v2/nnUNet_results/${dataset_name}/${nnunet_trainer}__nnUNetPlans__${configuration}/fold_${fold}/testZur --gt-folder ${nnUNet_raw}/${dataset_name}/labelsTsZur -dname sci-zurich-region

    echo "-------------------------------------------"
    echo "Running ANIMA evaluation on Colorado Test set"
    echo "-------------------------------------------"

    python testing/compute_anima_metrics.py --pred-folder ~/nnunet-v2/nnUNet_results/${dataset_name}/${nnunet_trainer}__nnUNetPlans__${configuration}/fold_${fold}/testCol --gt-folder ${nnUNet_raw}/${dataset_name}/labelsTsCol -dname sci-colorado-region

done
