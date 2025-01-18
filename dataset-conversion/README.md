## Data

The data used for training the model are hosted on a private repository according to the [BIDS](https://bids.neuroimaging.io) standard. They are gathered from three different sites as shown below (in brackets: the name of the dataset at NeuroPoly's internal server):

- University of Zurich 🇨🇭
  - Traumatic SCI
    - Lesion etiology: Mix of traumatic and ischemic lesions
    - Surgery: Mix of operated and non-operated subjects
    - Contrasts available: Sagittal T1w, Sagittal T2w, Axial T2w
    - Contrasts used for training: Sagittal T2w
    - Manual segmentations for both spinal cord (SC) and lesion only available for Sagittal T2w
    - Mix of single and multiple (up to 3) sessions
    - Number of subjects: 97 (`sci-zurich`)
  - Non-traumatic SCI
    - Lesion etiology: Degenerative Cervical Myelopathy (DCM)
    - Contrasts used for training: Axial T2w
    - Manual segmentations lesions only available for Axial T2w
      - SC segmentations generated with `sct_deepseg_sc` and manually corrected
    - Single session
    - Number of subjects: 57 
      - 14 (`dcm-zurich-lesions`) (training only)
      - 43 (`dcm-zurich-lesions-20231115`)
- University of Colorado Anschutz Medical Campus (`sci-colorado`) 🇺🇸
  - Contrasts available: Axial T1w, Axial T2w
  - Contrasts used for training: Axial T2w
  - Manual segmentations for both SC and lesion only available for Axial T2w
  - Single session
  - Number of subjects: 80
  - Post-operative subjects only
- University of Paris (`sci-paris`) 🇫🇷
  - Contrasts used for training: Sagittal T2w
  - Manual segmentations for both SC and lesion only available for Sagittal T2w
  - Single session
  - Number of subjects: 14 (training only)
  - Lesion etiology: traumatic lesions
  - Surgery: Mix of operated and non-operated subjects
- PRAXIS Spinal Cord Institue 🇨🇦
  - Sites: `site-003` (testing only), `site-012` (training only), `site-013` (training only), `site-014` (testing only)
  - Contrasts used for training: Sagittal T2w
  - Manual segmentations for both spinal cord (SC) and lesion available for Sagittal T2w
  - Single session
  - Number of subjects (total): 84
  - Lesion etiology: Acute pre-operative traumatic SCI
  


### Names and Versions of the Datasets

- `sci-zurich`
  - Name: git@data.neuro.polymtl.ca:datasets/sci-zurich
  - Commit: 3ba898e0  (can be checked by running `git log --oneline` after downloading the dataset)
- `sci-colorado`
  - Name: git@data.neuro.polymtl.ca:datasets/sci-colorado
  - Commit: 558303f
- `sci-paris`   (not used for training)
  - Name: git@data.neuro.polymtl.ca:datasets/sci-paris
  - Commit: 0a0d252
- `dcm-zurich-lesions`
  - Name: git@data.neuro.polymtl.ca:datasets/dcm-zurich-lesions
  - Commit: d214e06
- `dcm-zurich-lesions-20231115`
  - Name: git@data.neuro.polymtl.ca:datasets/dcm-zurich-lesions-20231115
  - Commit: 28a70d9
- `site-003` (PRAXIS)
  - Name: gitea@spineimage.ca:TOH/site_03.git
  - Commit: 92ee944
- `site-012` (PRAXIS)
  - Name: gitea@spineimage.ca:NSHA/site_012.git
  - Commit: 375059b
- `site-013` (PRAXIS)
  - Name: gitea@spineimage.ca:HGH/site_013.git
  - Commit: 7cd7611
- `site-014` (PRAXIS)
  - Name: gitea@spineimage.ca:HEJ/site_014.git
  - Commit: bd8099e


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
python convert_bids_to_nnUNetv2_all_sci_data.py --path-data ~/datasets/sci-zurich ~/datasets/sci-colorado ...
          --path-out ${nnUNet_raw} -dname tSCICombinedRegion -dnum 275 --split 0.8 0.2 --seed 50 --region-based
```

This command takes as inputs the list of datasets, the output path to store the converted data, the dataset name, the 
dataset number, the ratio in which train and test subjects are split, the random seed, 
and `--region-based` or `--multichannel` flag.
