## Data

The data used for training the model are hosted on a private repository according to the [BIDS](https://bids.neuroimaging.io) standard. They are gathered from three different sites as shown below (in brackets: the name of the dataset at NeuroPoly's internal server):

- University of Zurich (`sci-zurich`) 🇨🇭
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


### Names and Versions of the Datasets

- `sci-zurich`
  - Name: git@data.neuro.polymtl.ca:datasets/sci-zurich
  - Commit: b3cb6f51  (can be checked by running `git log --oneline` after downloading the dataset)
- `sci-colorado`
  - Name: git@data.neuro.polymtl.ca:datasets/sci-colorado
  - Commit: 1518ecd
- `sci-paris`
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
