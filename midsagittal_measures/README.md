# Midsagittal lesion morphometrics

This folder contains scripts to compare the midsagittal lesion length, lesion width, and tissue bridges obtained using different methods:

1. **_manual_**: manual measurements provided by Dario and Lynn
2. **_automatic_**: automatic measurement computed from the spinal cord and lesion segmentations obtained using SCIsegV2 (i.e., SCT's `sct_deepseg lesion_sci_t2` + `sct_analyze_lesion`)

## 0. Download the datasets

```console
git clone git@data.neuro.polymtl.ca:datasets/sci-zurich
cd sci-zurich
git annex init
git annex dead here
git annex get $(find . -name "*sag*T2*")

git clone git@data.neuro.polymtl.ca:datasets/nisci-trial
cd nisci-trial
git annex init
git annex dead here
git annex get $(find . -name "*sag*T2*")
```

## 1. Compute midsagittal lesion length and width

Compute the midsagittal lesion length, lesion width, and tissue bridges using the SCT's `sct_analyze_lesion` function using the 
`01_compute_midsagittal_lesion_length_and_width.sh` script.
The script is run using the `sct_run_batch` wrapper script to process subjects in parallel.
Note that the script requires SCT v7.0 or higher and is designed to be run on GPU.

```bash
sct_run_batch -config config-01_compute_midsagittal_lesion_length_and_width.json
```

Note: the script is run separately for each dataset (`sci-zurich` and `nisci-trial`).

## 2. Aggregate lesion metrics across subjects

As the `01_compute_midsagittal_lesion_length_and_width.sh` script calls `sct_analyze_lesion` function, it outputs one XLS file per subject.
The XLS files are saved in the `/results` directory.
To make it easier to work, I read the XLS files and save the data in a CSV file using the `02_read_xls_files.py` script.

```bash
python 02_combine_xlsx_files.py -dir ~/results/sci-zurich/midsagittal_measures_2025-10-20/results
python 02_combine_xlsx_files.py -dir ~/results/nisci-trial/midsagittal_measures_2025-10-20/results
```

## 3. Generate figures

Finally, we generate figures:

- `03_generate_lesion_metric_plots.py`: generates scatter plots and Bland-Altman plots between manual and automatic lesion metrics.
- `04_generate_trajectory_plots.py`: generates trajectory plots of clinical scores over time. Participants are grouped based on cutoffs of baseline lesion  metrics obtained from URP-CTREE analysis.
- `05_descriptive_analysis.py`: generates 2x2 figure showing demographic data (sex, age, AIS grades over time, and injury levels).

Please see the scripts' descriptions for example usages.