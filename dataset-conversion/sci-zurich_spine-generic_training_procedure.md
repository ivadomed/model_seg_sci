### Training procedure for the combined SCI-Zurich & Spine-Generic dataset

This document describes the procedure for training an nnUNet model on the combined SCI-Zurich and Augmented Spine-Generic dataset as it is slightly different from the standard nnUNet training procedure. 

The main difference comes from the fact that we are using the Spine-Generic dataset solely for offline data augmentation. In short, the augmentation is done by randomly picking a lesion from a SCI-Zurich subject and placing on the spinal cord of Spine-Generic subject. Hence, any of the SCI-Zurich subjects involved in the augmentation _cannot_ appear in the validation and the test set of the actual combined dataset. 

#### Step 1: Conversion of the datasets

Use the following command to combine both the datasets into a single dataset:

```bash
python convert_sci-zurich_spine-generic_to_nnUNetv2.py --path-data ~/datasets/sci-zurich-rpi ~/datasets/Dataset551_SpineGenericMultiSubject/imagesTr --path-out ${nnUNet_raw} -dname tSCIZurichAugSpineG100Resample -dnum 551 --seed 99 --split 0.6 0.2 0.2
```

Note that the first dataset here is the BIDS-formatted SCI-Zurich dataset and the second dataset is the Augmented Spine Generic already in the nnUNet format. **Importantly**, this also creates a splits_final.json file that manually splits the Zurich subjects into appropriate train and validation splits. The splits file has to be created manually because the Zurich subjects used for augmenting lesions cannot leak into the validation set. 

#### Step 2: Preprocessing

Run the usual preprocessing step: 

```bash
nnUNetv2_plan_and_preprocess -d 551 --verify_dataset_integrity
```

**WARNING**: Before moving on to training, copy the `splits_final.json` to the preprocessed dataset folder. This instructs nnUNet to use the manual splits instead of creating the default 5-fold cross-validation splits.


#### Step 3: Training and Inference

For training, run:

```bash
nnUNetv2_train 551 3d_fullres 0
```

This runs the model on fold 0 (i.e. the only fold that exists). 

For inference, run:

```bash
nnUNetv2_predict -i ${nnUNet_raw}/Dataset551_tSCIZurichAugSpineG100Resample/imagesTs -o ~/nnunet-v2/nnUNet_results/Dataset551_tSCIZurichAugSpineG100Resample/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/test -d 551 -f 0 -c 3d_fullres
```

This creates a new folder `test` inside the `fold_0` folder, which contains the predictions for the test set.


#### Optional: Training on multiple seeds

With the current setup, each (offline-augmented) dataset is created with a specific seed (in this case, `seed=99` was chosen). For training on other seeds: 

1. Create the augmented spine-generic dataset with the new seed 
2. Repeat Steps 1-3 above