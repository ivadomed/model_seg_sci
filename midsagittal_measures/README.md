# Midsagittal lesion length and width

This folder contains scripts to compare the midsagittal lesion length and width obtained using different methods:

1. **_manual_**: manual measurement provided by collaborators
2. **_automatic_SCIsegV2_**: the midsagittal lesion length and width are computed from the spinal cord and lesion segmentations obtained using SCIsegV2

## 0. Download the dataset

```console
git clone git@data.neuro.polymtl.ca:datasets/sci-zurich
cd sci-zurich
git annex init
git annex dead here
git annex get $(find . -name "*sag*T2*")
```

## 1. Compute midsagittal lesion length and width

Compute the midsagittal lesion length and width using the SCT's `sct_analyze_lesion` function using the 
`01_compute_midsagittal_lesion_length_and_width.sh` script.
The script is run using the `sct_run_batch` wrapper script to process subjects in parallel.
Note that the script requires SCT v6.4 or higher and is designed to be run on GPU.

```bash
sct_run_batch -config config-01_compute_midsagittal_lesion_length_and_width.json
```

NOTE: the script is run several times for different SCT branches to compare different versions of the `sct_analyze_lesion` function.

## 2. Aggregate lesion metrics across subjects

As the `01_compute_midsagittal_lesion_length_and_width.sh` script calls `sct_analyze_lesion` function, it outputs one XLS file per subject.
The XLS files are saved in the `/results` directory.
To make it easier to work, I read the XLS files and save the data in a CSV file using the `02_read_xls_files.py` script.

```bash
python 02_read_xls_files.py -dir <DIR_NAME>/results -branch master -pred-type GT
python 02_read_xls_files.py -dir <DIR_NAME>/results -branch master -pred-type SCIsegV2
python 02_read_xls_files.py -dir <DIR_NAME>/results -branch PR4656 -pred-type GT
python 02_read_xls_files.py -dir <DIR_NAME>/results -branch PR4656 -pred-type SCIsegV2
```