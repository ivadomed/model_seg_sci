## Training and Testing the Model

If you have made it here, this means that you are interested in training the model! This documentation provides details on how to train the model. Training with nnUNet is simple. Once the environment is properly configured, the following steps should get the model up and running:

### Step 1: Configuring the environment

1. Create a conda environment with the following command:
```
conda create -n venv_nnunet python=3.9
```

2. Activate the environment with the following command:
```
conda activate venv_nnunet
```

3. Install the required packages with the following command:
```
pip install -r requirements.txt
```

> **Note**
> The `requirements.txt` does not install nnUNet. It has to be installed separately. See [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md) for installation instructions. Please note that the nnUNet version used in this work is `58a6c40`. 

#### Step 2: Verifying dataset integrity

This section assumes that the installation was successful and the nnUNet-related environment variables have been set.

```bash
nnUNetv2_plan_and_preprocess -d <dataset-num> --verify_dataset_integrity -c 3d_fullres
```
This command first verifies whether the dataset format is correct. Then, an internal pre-processing is automatically done to automatically set the model (hyper-) parameters for training. Note that the `<dataset-num>` is the one that was used during the conversion to the nnUNet format (see Step 2.2).

#### Step 3: Training

```bash
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train <dataset-num> 3d_fullres 0 -tr nnUNetTrainer_2000epochs
```
This commands starts training a 3D model on `fold 0` of the dataset for 2000 epochs. The model can also be trained on all 5 folds. See `scripts` for an example of how to train on all folds in a (sequential) loop. 

#### Step 4: Testing

Because of how the dataset was converted, we have two test folders `testZur` and `testCol` consisting of subjects from these sites. Hence, we need to run the inference twice, once for each test folder.

```bash
# run inference on zurich test set
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i ${nnUNet_raw}/Dataset<dataset-num>_<dataset-name>/imagesTsZur -tr nnUNetTrainer_2000epochs -o ~/nnunet-v2/nnUNet_results/Dataset<dataset-num>_<dataset-name>/nnUNetTrainer_2000epochs__nnUNetPlans__3d_fullres/fold_0/testZur -d <dataset-num> -f 0 -c 3d_fullres
```

```bash
# run inference on colorado test set
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i ${nnUNet_raw}/Dataset<dataset-num>_<dataset-name>/imagesTsCol -tr nnUNetTrainer_2000epochs -o ~/nnunet-v2/nnUNet_results/Dataset<dataset-num>_<dataset-name>/nnUNetTrainer_2000epochs__nnUNetPlans__3d_fullres/fold_$fold/testCol -d <dataset-num> -f 0 -c 3d_fullres
```

#### Step 5: Evaluating the model performance

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
