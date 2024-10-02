# Midsagittal lesion length and width

This folder contains scripts to compare the midsagittal lesion length and width obtained using different methods:

1. **_manual_**: manual measurement provided by collaborators
2. **_automatic_SCIsegV2_**: the midsagittal lesion length and width are computed from the spinal cord and lesion segmentations obtained using SCIsegV2

## 1. Download the dataset

```console
git clone git@data.neuro.polymtl.ca:datasets/sci-zurich
cd sci-zurich
git annex init
git annex dead here
git annex get $(find . -name "*sag*T2*")
```

## 2. Compute midsagittal lesion length and width

Compute the midsagittal lesion length and width using the `01_compute_midsagittal_lesion_length_and_width.sh` script.
The script is run using the `sct_run_batch` wrapper script to process subjects in parallel.
Note that the script requires SCT v6.4 or higher and is designed to be run on GPU.

```bash
sct_run_batch -config config-01_compute_midsagittal_lesion_length_and_width.json
```