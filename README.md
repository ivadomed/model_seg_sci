# Automated Segmentation of the Spinal Cord and Hyperintense Lesions in Acute Spinal Cord Injury

This repository contains the code for deep learning-based segmentation of the spinal cord and T2-weighted hyperintense lesions in acute spinal cord injury (SCI). The code is based on the well-known [nnUNet framework](https://github.com/MIC-DKFZ/nnUNet).


## Model Overview

TODO: add a screenshot/gif of input, network and output


## Data

The data used for training the model are hosted on a private repository according to the [BIDS](https://bids.neuroimaging.io) standard. They are gathered from three different sites as shown below (in brackets: the name of the dataset at NeuroPoly's internal server):

- University of Zurich (`sci-zurich`) (N=) 🇨🇭
  - Contrasts available: Sagittal T1w, Sagittal T2w, Axial T2w
  - Contrasts used for training: Sagittal T2w
  - Manual segmentations for both spinal cord (SC) and lesion only available for Sagittal T2w
  - Mix of single and multiple (up to 3) sessions
  - Number of subjects: 97
- University of Colorado Anschutz Medical Campus (`sci-colorado`) 🇺🇸
  - Contrasts available: Axial T1w, Axial T2w
  - Contrasts used for training: Axial T2w
  - Manual segmentations for both SC and lesion only available for Axial T2w
  - Single session
  - Number of subjects: 80
- XXX (`sci-paris`) 🇫🇷
  - Contrasts available: Axial T2w
  - Contrasts used for training: Axial T2w
  - Manual segmentations for both SC and lesion available for Axial T2w
  - Single session
  - Number of subjects: 14


## Getting started

### Dependencies

TODO: add updated requirements.txt --> those are the dependencies

### Step 1: Cloning the Repository

~~~
git clone https://github.com/ivadomed/model_seg_sci.git
~~~

### Names and Versions of the Dataset

- `sci-zurich`
  - Name: git@data.neuro.polymtl.ca:datasets/sci-zurich
  - Commit: b3cb6f51  (can be checked by running `git log --oneline` after downloading the dataset)
- `sci-colorado`
  - Name: git@data.neuro.polymtl.ca:datasets/sci-colorado
  - Commit: 1518ecd
- `sci-paris`
  - Name: git@data.neuro.polymtl.ca:datasets/sci-paris
  - Commit: c4e3bf7

### Downloading the Dataset

The downloading procedure is same for all the three datasets, just replace the "<dataset-name>" by the name of the dataset you want to download from the server.

~~~
git clone git@data.neuro.polymtl.ca:datasets/<dataset-name>
cd <dataset-name>
git annex get .
cd ..
~~~
 
### Step 2: Preparing the Data

The best part about this model is that there is **no preprocessing required**! The model is directly trained on the raw data. The only data preparation step is to convert the data to the nnUNet format. The following commands are used for converting the dataset. 

#### Step 2.1: Reorientation

For uniformity across the multi-site data, all images are converted to RPI orientation as a first step. This is done using [Spinal Cord Toolbox](https://spinalcordtoolbox.com) (SCT). From your unix-based terminal, run the following command from the root directory of the downloaded dataset:

```bash
shopt -s globstar; for file in **/*.nii.gz;do sct_image -i ${file} -setorient RPI -o ${file}; done
``` 
This should do an in-place conversion of all the images (and labels) to RPI orientation.

#### Step 2.2: Conversion to nnUNet format

The next step is to convert the data to the nnUNet format. Run the following command:
```bash
python convert_bids_to_nnUNetv2_all_sci_data.py --path-data ~/datasets/sci-zurich-rpi ~/datasets/sci-colorado-rpi ~/datasets/sci-paris-rpi  --path-out ${nnUNet_raw} 
          -dname tSCICombinedRegion -dnum 275 --split 0.8 0.2 --seed 50 --region-based
```
This command takes as inputs the list of RPI-reoriented datasets, the output path to store the converted data, the dataset name, number, the ratio in which train and test subjects are split, the seed and the flag `--region-based` which used to create multiclass labels (SC and lesion).

> **Note**
> This assumes the nnUNet has been successfully installed and the necessary nnUNet-related environment variables have been set. Please refer to the [nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md) for more details.


### Step 3: Training and Testing the Model

Training with nnUNet is simple. Running the following commands should get the model up and running:

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

Once testing is done, the model's SC and lesion segmentation performance can be evaluated by computing some quantitative metrics. We use [ANIMA]()'s `animaSegPerfAnalyzer`
for this purpose. The following command can be used to compute the metrics:

```bash
# compute metrics for zurich test predictions
python testing/compute_anima_metrics.py --pred-folder ~/nnunet-v2/nnUNet_results/Dataset<dataset-num>_<dataset-name>/nnUNetTrainer_2000epochs__nnUNetPlans__3d_fullres/fold_0/testZur --gt-folder ${nnUNet_raw}/Dataset<dataset-num>_<dataset-name>/labelsTsZur -dname sci-zurich-region
```

```bash
# compute metrics for colorado test predictions
python testing/compute_anima_metrics.py --pred-folder ~/nnunet-v2/nnUNet_results/Dataset<dataset-num>_<dataset-name>/nnUNetTrainer_2000epochs__nnUNetPlans__3d_fullres/fold_0/testCol --gt-folder ${nnUNet_raw}/Dataset<dataset-num>_<dataset-name>/labelsTsCol -dname sci-colorado-region
```

### Only Inference

The instructions for running inference can be found [here]().


