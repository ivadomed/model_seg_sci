## Data

The data used for training the model are hosted on a private repository according to the [BIDS](https://bids.neuroimaging.io) standard. They are gathered from a single site as shown below (in brackets: the name of the dataset at NeuroPoly's internal server):

- Technical University of Munich (`mslesion-munich`) 
  - Contrasts available: Sagittal T2w, Axial T2w
  - Contrasts used for training: Axial T2w
  - Manual segmentations for both spinal cord (SC) and lesion only available for Axial T2w
  - Two longitudinal sessions per subject
  - Number of subjects: 337


### Names and Versions of the Datasets

- `mslesion-munich`
  - Name: git@data.neuro.polymtl.ca:datasets/TBD
  - Commit: TBD  (can be checked by running `git log --oneline` after downloading the dataset)


### Downloading the Data

~~~
git clone git@data.neuro.polymtl.ca:datasets/<dataset-name>
cd <dataset-name>
git annex get .
cd ..
~~~

### Preparing the Data

The best part about this model is that there is **no preprocessing required**! The model is directly trained on the raw data. The only data preparation step is to convert the data to the nnUNet format. The following commands are used for converting the dataset. 

#### Reorientation

For uniformity across the multi-site data, all images are converted to RPI orientation as a first step. This is done using [Spinal Cord Toolbox](https://spinalcordtoolbox.com) (SCT). From your unix-based terminal, run the following command from the root directory of the downloaded dataset:

```bash
shopt -s globstar; for file in **/*.nii.gz;do sct_image -i ${file} -setorient RPI -o ${file}; done
``` 

This should do an in-place conversion of all the images (and labels) to RPI orientation.

#### Conversion to nnUNet format

The next step is to convert the data to the nnUNet format. Run the following command:

```bash
python convert_bids_to_nnUNetv2_munich.py --path-data ~/raid3/Julian/data/segmentation_project/datasets_processing
          --path-out ${nnUNet_raw} -dname TUM_MS_Dataset_Stitched_BIDS_Region -dnum 901 --split 0.8 0.2 --seed 50 --region-based
```

This command takes as inputs the list of RPI-reoriented datasets, the output path to store the converted data, the dataset name, number, the ratio in which train and test subjects are split, the seed and the flag `--region-based` which used to create multiclass labels (SC and lesion).

> **Note**
> This assumes the nnUNet has been successfully installed and the necessary nnUNet-related environment variables have been set. Please refer to the [nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md) for more details.

#### Cropping around the spinal cord

The cropping of the input images, spinal cord GT, and lesion GT around the spinal cord segmentation can be done using 
the `crop_around_sc.sh` script. The cropping is done based on the dilated spinal cord GT.