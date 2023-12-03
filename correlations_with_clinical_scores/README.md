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
  "path_output" : "<PATH_TO_COMBINED_DATASET>_2023-08-18",
  "script"      : "<PATH_TO_REPO>/model_seg_sci/correlations_with_clinical_scores/01_analyze_lesions.sh",
  "jobs"        : 8,
  "script_args" : "<PATH_TO_REPO>/model_seg_sci/packaging/run_inference_single_subject.py <PATH_TO_MODEL>/sci-multisite-model_seed{XXX}"
}
```

ℹ️ `script_args` argument is used to pass arguments to the `comparison_with_other_methods_{sc,lesion}.sh` script. 
In this case, we pass the path to the `run_inference_single_subject.py` script and the path to the nnUNet model.

ℹ️ You can run the script across all train/test splits (i.e., seeds) by using the following command:

```console
for config_file in config_analyze_lesions_seed*.json;do echo sct_run_batch -config $config_file;done
```

⚠️ Make sure that the input dataset (`path_data`) corresponds with the seed of the nnUNet model (`script_args`).
