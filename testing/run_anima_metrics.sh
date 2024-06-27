#!/bin/bash
# Training nnUNet on just the Zurich dataset


dataset_name="Dataset502_allSCIsegV2MultichannelSeed710"
# training_type="region-based"
training_type="multi-channel"

nnunet_trainers=("nnUNetTrainer" "nnUNetTrainerDA5")
# nnunet_trainers=("nnUNetTrainerDA5_DiceCELoss_noSmooth")
# nnunet_trainers=("nnUNetTrainerDA5")

configuration="3d_fullres"

folds=(0 1 2 3 4)

test_sites=("dcm-zurich-lesions-20231115" "sci-colorado" "sci-zurich" "site-003" "site-014")


for nnunet_trainer in ${nnunet_trainers[@]}; do

    for fold in ${folds[@]}; do

        for site in ${test_sites[@]}; do
            echo "-------------------------------------------"
            echo "Running ANIMA evaluation on trainer, fold, site: ${nnunet_trainer}, ${fold}, ${site}"
            echo "-------------------------------------------"
            python /home/GRAMES.POLYMTL.CA/u114716/sci_seg/model_seg_sci/testing/compute_anima_metrics.py --pred-folder ~/nnunet-v2/nnUNet_results/${dataset_name}/${nnunet_trainer}__nnUNetPlans__${configuration}/fold_${fold}/test_${site} --gt-folder ${nnUNet_raw}/${dataset_name}/labelsTs_${site} --training-type ${training_type}
            
            # python /home/GRAMES.POLYMTL.CA/u114716/sci_seg/model_seg_sci/testing/compute_anima_metrics.py --pred-folder ~/nnunet-v2/nnUNet_results/${dataset_name}/${nnunet_trainer}__nnUNetPlans__${configuration}/fold_${fold}/test_${site} --gt-folder /scratch/SCIsegv2/training_datasets/${dataset_name}/labelsTs_${site} --training-type ${training_type}
        done

    done

done
