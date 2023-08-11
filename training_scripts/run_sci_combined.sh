#!/bin/bash
# Training nnUNet on just the Zurich dataset

# echo "-------------------------------------------------------"
# echo "Running preprocessing and verifying dataset integrity"
# echo "-------------------------------------------------------"
# nnUNetv2_plan_and_preprocess -d 275 --verify_dataset_integrity -c 3d_fullres


# folds=(2 3 4)
folds=(0)

for fold in ${folds[@]}; do
    echo "-------------------------------------------"
    echo "Training on Fold $fold"
    echo "-------------------------------------------"
    
    # training
    CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 275 3d_fullres $fold -tr nnUNetTrainer_2000epochs

    echo ""
    echo "-------------------------------------------"
    echo "Training completed, Testing on Fold $fold"
    echo "-------------------------------------------"

    # run inference on zurich test set
    CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i ${nnUNet_raw}/Dataset275_tSCICombined/imagesTsZur -tr nnUNetTrainer_2000epochs -o ~/nnunet-v2/nnUNet_results/Dataset275_tSCICombined/nnUNetTrainer_2000epochs__nnUNetPlans__3d_fullres/fold_$fold/testZur -d 275 -f $fold -c 3d_fullres

    # run inference on colorado test set
    CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i ${nnUNet_raw}/Dataset275_tSCICombined/imagesTsCol -tr nnUNetTrainer_2000epochs -o ~/nnunet-v2/nnUNet_results/Dataset275_tSCICombined/nnUNetTrainer_2000epochs__nnUNetPlans__3d_fullres/fold_$fold/testCol -d 275 -f $fold -c 3d_fullres

    echo "-------------------------------------------"
    echo "Running ANIMA evaluation on Zurich Test set"
    echo "-------------------------------------------"

    python testing/compute_anima_metrics.py --pred-folder ~/nnunet-v2/nnUNet_results/Dataset275_tSCICombined/nnUNetTrainer_2000epochs__nnUNetPlans__3d_fullres/fold_$fold/testZur --gt-folder ${nnUNet_raw}/Dataset275_tSCICombined/labelsTsZur -dname sci-zurich-region

    echo "-------------------------------------------"
    echo "Running ANIMA evaluation on Colorado Test set"
    echo "-------------------------------------------"

    python testing/compute_anima_metrics.py --pred-folder ~/nnunet-v2/nnUNet_results/Dataset275_tSCICombined/nnUNetTrainer_2000epochs__nnUNetPlans__3d_fullres/fold_$fold/testCol --gt-folder ${nnUNet_raw}/Dataset275_tSCICombined/labelsTsCol -dname sci-colorado-region

done
