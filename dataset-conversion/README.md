## Data

The data used for training the model are hosted on a private repository according to the [BIDS](https://bids.neuroimaging.io) standard. They are gathered from three different sites as shown below (in brackets: the name of the dataset at NeuroPoly's internal server):

- University of Zurich (`sci-zurich`) 🇨🇭
  - Contrasts available: Sagittal T1w, Sagittal T2w, Axial T2w
  - Contrasts used for training: Sagittal T2w
  - Manual segmentations for both spinal cord (SC) and lesion only available for Sagittal T2w
  - Mix of single and multiple (up to 3) sessions
  - Number of subjects: 97
  - Lesion etiology: Mix of traumatic and ischemic lesions
  - Surgery: Mix of operated and non-operated subjects
- University of Colorado Anschutz Medical Campus (`sci-colorado`) 🇺🇸
  - Contrasts available: Axial T1w, Axial T2w
  - Contrasts used for training: Axial T2w
  - Manual segmentations for both SC and lesion only available for Axial T2w
  - Single session
  - Number of subjects: 80
  - Post-operative subjects only


### Names and Versions of the Datasets

- `sci-zurich`
  - Name: git@data.neuro.polymtl.ca:datasets/sci-zurich
  - Commit: b3cb6f51  (can be checked by running `git log --oneline` after downloading the dataset)
- `sci-colorado`
  - Name: git@data.neuro.polymtl.ca:datasets/sci-colorado
  - Commit: 1518ecd
- `sci-paris`   (not used for training)
  - Name: git@data.neuro.polymtl.ca:datasets/sci-paris
  - Commit: c4e3bf7

### Downloading the Data

The downloading procedure is same for all the three datasets, just replace the "<dataset-name>" by the name of the dataset you want to download from the server.

~~~
git clone git@data.neuro.polymtl.ca:datasets/<dataset-name>
cd <dataset-name>
git annex get .
cd ..
~~~

### Preparing the Data

The best part about this model is that there is **no preprocessing required**! The model is directly trained on the raw data. The only data preparation step is to convert the data to the nnUNet format. The following commands are used for converting the dataset. 


#### Conversion to nnUNet format

The next step is to convert the data to the nnUNet format using the `convert_bids_to_nnUNetv2_all_sci_data.py` script.

The script supports both region-based and multichannel labels. 

Note: the script performs RPI reorientation of the images and labels; i.e., there is no need to reorient the images and 
labels before running the script.

Note: create venv and install `dataset-conversion/requirements.txt` before running the script.

```bash
python convert_bids_to_nnUNetv2_all_sci_data.py --path-data ~/datasets/sci-zurich-rpi ~/datasets/sci-colorado-rpi
          --path-out ${nnUNet_raw} -dname tSCICombinedRegion -dnum 275 --split 0.8 0.2 --seed 50 --region-based
```

This command takes as inputs the list of datasets, the output path to store the converted data, the dataset name, the 
dataset number, the ratio in which train and test subjects are split, the random seed, 
and `--region-based` or `--multichannel` flag.

> **Note**
> This assumes the nnUNet has been successfully installed and the necessary nnUNet-related environment variables have been set. Please refer to the [nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md) for more details.

#### Cropping around the spinal cord

The cropping of the input images, spinal cord GT, and lesion GT around the spinal cord segmentation can be done using 
the `crop_around_sc.sh` script. The cropping is done based on the dilated spinal cord GT.