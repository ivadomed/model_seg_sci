#!/bin/bash
# Training nnUNet on all SCI datasets


# define arguments for nnUNet
dataset_name="Dataset501_allSCIsegV2RegionSeed710"
dataset_num="501"
# nnunet_trainer="nnUNetTrainerDiceCELoss_noSmooth"         # default: nnUNetTrainer or nnUNetTrainer_2000epochs
nnunet_trainer="nnUNetTrainerDA5"                           # trainer variant with aggressive data augmentation
# nnunet_trainer="nnUNetTrainerDA5_DiceCELoss_noSmooth"       # custom trainer

# NOTE: when using the standard model, use the following nnunet_plans_file
nnunet_plans_file="nnUNetPlans"

configurations=("3d_fullres")                               # for 2D training, use "2d"
cuda_visible_devices=0
fold=0
test_sites=("site-01-dcm" "site-01-tSCI" "site-02" "site-003" "site-014")

# define final variables where the data will be copied
final_prepro_dir="/home/$(whoami)/projects/def-prof/$(whoami)/datasets/nnUNet_preprocessed"
final_results_dir="/home/$(whoami)/code/nnunet-v2/nnUNet_results"

echo $SLURM_TMPDIR

echo "-------------------------------------------"
echo "Moving the dataset to SLURM_TMPDIR: ${SLURM_TMPDIR}"
echo "-------------------------------------------"

# create folders in SLURM_TMPDIR
if [[ ! -d $SLURM_TMPDIR/nnUNet_raw ]]; then
    mkdir $SLURM_TMPDIR/nnUNet_raw

    # copy the dataset to SLURM_TMPDIR
    cp -r /home/$(whoami)/projects/def-prof/$(whoami)/datasets/nnUNet_raw/${dataset_name} ${SLURM_TMPDIR}/nnUNet_raw
fi

# create folders in SLURM_TMPDIR
mkdir $SLURM_TMPDIR/nnUNet_preprocessed
mkdir $SLURM_TMPDIR/nnUNet_results

# temporarily export the nnUNet environment variables
export nnUNet_raw=$SLURM_TMPDIR/nnUNet_raw
export nnUNet_preprocessed=$SLURM_TMPDIR/nnUNet_preprocessed
export nnUNet_results=$SLURM_TMPDIR/nnUNet_results


echo "-------------------------------------------------------"
echo "Running preprocessing and verifying dataset integrity"
echo "-------------------------------------------------------"
nnUNetv2_plan_and_preprocess -d ${dataset_num} -c ${configurations} --verify_dataset_integrity



for configuration in ${configurations[@]}; do
    echo "-------------------------------------------"
    echo "Training on Fold $fold, Configuration $configuration"
    echo "-------------------------------------------"

    # training
    CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_train ${dataset_num} $configuration $fold -tr ${nnunet_trainer} -p ${nnunet_plans_file}

    echo ""
    echo "-------------------------------------------"
    echo "Training completed, Testing on Fold $fold"
    echo "-------------------------------------------"

    for site in ${test_sites[@]}; do
        echo "-------------------------------------------"
        echo "Testing on site: $site"
        echo "-------------------------------------------"
        # run inference on test set
        CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_predict -i ${nnUNet_raw}/${dataset_name}/imagesTs_${site} -tr ${nnunet_trainer} -p ${nnunet_plans_file} -o ${nnUNet_results}/${dataset_name}/${nnunet_trainer}__${nnunet_plans_file}__${configuration}/fold_${fold}/test_${site} -d ${dataset_num} -f $fold -c ${configuration} # -step_size 0.9 --disable_tta
    done

done

echo ""
echo "--------------------------------------------------------------------------------------------"
echo "Testing done, Moving the results/preprocessed data from ${SLURM_TMPDIR} to the home directory"
echo "-----------------------------------------------------------------------------------------"


# copy the results back to the home directory
cp -r ${SLURM_TMPDIR}/nnUNet_results/${dataset_name} ${final_results_dir}

# echo ""
echo "-------------------------------------------"
echo "Activating Metrics Environment ..."
echo "-------------------------------------------"
source /home/$(whoami)/envs/venv_metrics/bin/activate


for site in ${test_sites[@]}; do

    echo "-------------------------------------------"
    echo "Running Metrics Reloaded on site $site ..."
    echo "-------------------------------------------"

    # compute metrics
    python /home/$(whoami)/code/MetricsReloaded/compute_metrics_reloaded.py \
        -reference ${nnUNet_raw}/${dataset_name}/labelsTs_${site} \
        -prediction ${nnUNet_results}/${dataset_name}/${nnunet_trainer}__${nnunet_plans_file}__${configuration}/fold_${fold}/test_${site} \
        -output ${nnUNet_results}/${dataset_name}/${nnunet_trainer}__${nnunet_plans_file}__${configuration}/fold_${fold}/test_${site}/${site}_metrics.csv \
        -metrics dsc nsd rel_vol_error lesion_ppv lesion_sensitivity lesion_f1_score lcwa ref_count pred_count \
        -jobs 8

    # copy the results csv back to the home directory
    cp -r ${nnUNet_results}/${dataset_name}/${nnunet_trainer}__${nnunet_plans_file}__${configuration}/fold_${fold}/test_${site}/*.csv ${final_results_dir}/${dataset_name}/${nnunet_trainer}__${nnunet_plans_file}__${configuration}/fold_${fold}/test_${site}

done

echo "-------------------------------------------"
echo "Metrics computation done!"
echo "-------------------------------------------"


# # Running inference using SCIsegV1 model
# echo "-------------------------------------------"
# echo "Running inference using SCIsegV1 model"
# echo "-------------------------------------------"

# inference_dataset_name="Dataset403_tSCI3SitesALPhase3Seed710"
# mkdir $SLURM_TMPDIR/nnUNet_results/${inference_dataset_name}

# cp -r /home/$(whoami)/code/nnunet-v2/nnUNet_results/${inference_dataset_name}/nnUNetTrainerDiceCELoss_noSmooth__nnUNetPlans__3d_fullres ${SLURM_TMPDIR}/nnUNet_results/${inference_dataset_name}/nnUNetTrainerDiceCELoss_noSmooth__nnUNetPlans__3d_fullres

# for site in ${test_sites[@]}; do
#     echo "-------------------------------------------"
#     echo "Testing SCISegV1 on site: $site"
#     echo "-------------------------------------------"
#     # run inference on test set
#     CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_predict -i ${nnUNet_raw}/${dataset_name}/imagesTs_${site} -tr nnUNetTrainerDiceCELoss_noSmooth -p nnUNetPlans -o ${nnUNet_results}/${dataset_name}/${nnunet_trainer}__${nnunet_plans_file}__${configuration}/fold_${fold}/inference_scisegv1/test_${site} -d ${inference_dataset_name} -f 1 -c 3d_fullres
# done

# echo ""
# echo "--------------------------------------------------------------------------------------------"
# echo "Testing done, Moving the results/preprocessed data from ${SLURM_TMPDIR} to the home directory"
# echo "-----------------------------------------------------------------------------------------"

# # copy the inference results back to the home directory
# cp -r ${SLURM_TMPDIR}/${nnUNet_results}/${dataset_name}/${nnunet_trainer}__${nnunet_plans_file}__${configuration}/fold_${fold}/inference_scisegv1 ${final_results_dir}/${dataset_name}/${nnunet_trainer}__${nnunet_plans_file}__${configuration}/fold_${fold}/inference_scisegv1


echo "-------------------"
echo "Job Done!"
echo "-------------------"
