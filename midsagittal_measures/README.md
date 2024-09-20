# Midsagittal lesion length and width

This folder contains scripts to compare the midsagittal lesion length and width obtained using different methods:

1. **_manual_**: manual measurement provided by collaborators
2. **_automatic_SCIsegV2_**: the midsagittal lesion length and width are computed from the spinal cord and lesion segmentations obtained using SCIsegV2
3. **_automatic_contrast-agnostic_**: the midsagittal lesion length and width are computed from the spinal cord obtained using the contrast-agnostic model v2.4 and lesion segmentation obtained using SCIsegV2

The rationale for trying method 3 (contrast-agnostic model for SC seg) is that SCIsegV2 might not segment the spinal cord for some lateral slices (probably due to the presence of CSF and PVE). 
These missing border slices might then influence the estimation of the midsagittal slice (because the midsagittal slice is computed from the spinal cord segmentation; see code [here](https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/scripts/sct_analyze_lesion.py#L899)).
However, the `contrast-agnostic` model fails for some lumbar images with metal artifacts (see comments [here](https://github.com/ivadomed/model_seg_sci/pull/96)). So maybe we will not be able to use this method for all subjects.

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