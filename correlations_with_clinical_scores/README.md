## 1. Running the `01_analyze_lesions.sh` script

This section explains how to run the `01_analyze_lesions.sh` script across all subjects and across multiple seeds (each
seed corresponds to a different train/test split of the dataset).

The `analyze_lesions.sh` calls the SCT's `sct_analyze_lesion` function on:
- GT lesion and spinal cord segmentations (located under `derivatives/labels`)
- predicted lesion and SC segmentations (using our nnUNet SCIseg 3D model)

Outputs are saved under the `/results` directory (corresponding to the `${PATH_RESULTS}` variable retrieved from the 
caller `sct_run_batch`).

Example:

```console
sct_run_batch -config config_analyze_lesions_seed{XXX}.json
```

Example of the `config_analyze_lesions_seed{XXX}.json` file:

```json
{
  "path_data"   : "<PATH_TO_COMBINED_DATASET>_seed{XXX}",
  "path_output" : "<PATH_TO_COMBINED_DATASET>_analyze_lesions_seed{XXX}_2023-XX-XX",
  "script"      : "<PATH_TO_REPO>/model_seg_sci/correlations_with_clinical_scores/01_analyze_lesions.sh",
  "jobs"        : 8,
  "script_args" : "<PATH_TO_REPO>/model_seg_sci/packaging/run_inference_single_subject.py <PATH_TO_MODEL>/sci-multisite-model_seed{XXX}"
}
```

ℹ️ `script_args` argument is used to pass arguments to the `01_analyze_lesions.sh` script. 
In this case, we pass the path to the `run_inference_single_subject.py` script and the path to the nnUNet model.

ℹ️ You can run the script across all train/test splits (i.e., seeds) by using the following command (assuming you have 
5 different config files, one for each seed):

```console
for config_file in config_analyze_lesions_seed*.json;do echo sct_run_batch -config $config_file;done
```

⚠️ Make sure that the input dataset (`path_data`) corresponds with the seed of the nnUNet model (`script_args`).

## 2a. Generate regression plots using the `02a_generate_regplot_manual_vs_predicted.py` script

Generate sns.regplot for each metric (volume, length, max_axial_damage_ratio) manual vs SCIseg 3D lesion
segmentation BEFORE and AFTER active learning.

## 2b. Generate regression plots using the `02b_correlate_lesions_with_clinical_scores.py` script

Generate sns.lmplot for each metric (volume, length, max_axial_damage_ratio) and clinical score (AIS, LEMS, light touch,
pinprick) at initial, discharge and diff time points manual vs SCIseg 3D lesion segmentation.

3x3 mosaic of plots for each metric and clinical score can be then generated using `convert` command:

```console
convert volume_regplot_discharge_pin_prick_total.png volume_regplot_discharge_light_touch_total.png volume_regplot_discharge_LEMS.png +append row1.png
convert length_regplot_discharge_pin_prick_total.png length_regplot_discharge_light_touch_total.png length_regplot_discharge_LEMS.png +append row2.png
convert max_axial_damage_ratio_regplot_discharge_pin_prick_total.png max_axial_damage_ratio_regplot_discharge_light_touch_total.png max_axial_damage_ratio_regplot_discharge_LEMS.png +append row3.png
convert row1.png row2.png row3.png -append combined.png
```