## Training and Testing the Model

Training with nnUNet is simple. Running the commands in the following steps should get the model up and running:

#### Step 3.1: Verifying dataset integrity

```bash
nnUNetv2_plan_and_preprocess -d <dataset-num> --verify_dataset_integrity -c 3d_fullres
```
This command first verifies whether the dataset format is correct. Then, an internal pre-processing is automatically done to automatically set the model (hyper-) parameters for training. Note that the `<dataset-num>` is the one that was used during the conversion to the nnUNet format (see Step 2.2).

#### Step 3.2: Training

```bash
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train <dataset-num> 3d_fullres 0 -tr nnUNetTrainer_2000epochs
```
This commands starts training a 3D model on `fold 0` of the dataset for 2000 epochs. The model can also be trained on all 5 folds. See `scripts` for an example of how to train on all folds in a (sequential) loop. 

#### Step 3.3: Testing

Because of how the dataset was converted, we have two test folders `testZur` and `testCol` consisting of subjects from these sites. Hence, we need to run the inference twice, once for each test folder.

```bash
# run inference on zurich test set
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i ${nnUNet_raw}/Dataset<dataset-num>_<dataset-name>/imagesTsZur -tr nnUNetTrainer_2000epochs -o ~/nnunet-v2/nnUNet_results/Dataset<dataset-num>_<dataset-name>/nnUNetTrainer_2000epochs__nnUNetPlans__3d_fullres/fold_0/testZur -d <dataset-num> -f 0 -c 3d_fullres
```

```bash
# run inference on colorado test set
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i ${nnUNet_raw}/Dataset<dataset-num>_<dataset-name>/imagesTsCol -tr nnUNetTrainer_2000epochs -o ~/nnunet-v2/nnUNet_results/Dataset<dataset-num>_<dataset-name>/nnUNetTrainer_2000epochs__nnUNetPlans__3d_fullres/fold_$fold/testCol -d <dataset-num> -f 0 -c 3d_fullres
```

#### Step 3.4: Evaluating the model performance

Once testing is done, the model's SC and lesion segmentation performance can be evaluated by computing some quantitative metrics. We use [ANIMA](https://anima.readthedocs.io/en/latest/segmentation.html)'s `animaSegPerfAnalyzer` for this purpose. The following command can be used to compute the metrics:

```bash
# compute metrics for zurich test predictions
python testing/compute_anima_metrics.py --pred-folder ~/nnunet-v2/nnUNet_results/Dataset<dataset-num>_<dataset-name>/nnUNetTrainer_2000epochs__nnUNetPlans__3d_fullres/fold_0/testZur --gt-folder ${nnUNet_raw}/Dataset<dataset-num>_<dataset-name>/labelsTsZur -dname sci-zurich-region
```

```bash
# compute metrics for colorado test predictions
python testing/compute_anima_metrics.py --pred-folder ~/nnunet-v2/nnUNet_results/Dataset<dataset-num>_<dataset-name>/nnUNetTrainer_2000epochs__nnUNetPlans__3d_fullres/fold_0/testCol --gt-folder ${nnUNet_raw}/Dataset<dataset-num>_<dataset-name>/labelsTsCol -dname sci-colorado-region
```

All the above commands can be run from a single bash script. See `run_sci_combined.sh` for an example.
