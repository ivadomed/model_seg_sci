
## 01. Inference using the SCIseg nnUNet model

`01_run_inference_praxis.sh` is used to run inference using the SCIseg nnUNet model on any PRAXIS dataset. The script 
has to be run separately for each site.
The script segments both the spinal cord and SCI lesions:

### Spinal cord

The predicted spinal cords are visually inspected, manually corrected if necessary (using [manual_correction.py](https://github.com/spinalcordtoolbox/manual-correction/blob/main/manual_correction.py)), 
added to the BIDS dataset under `derivatives/labels`, and pushed to the repository.

### SCI lesions

The lesions predicted by the SCIseg model are compared against the manual GT lesions (located under `derivatives/labels`) 
using ANIMA segmentation performance metrics. The SCIseg lesion segmentation showed low lesion segmentation performance.

## 02. Training pre-operative nnUNet model from scratch

`02_run_training_praxis.sh` was used to train the pre-operative SCIseg nnUNet model. The model was trained on sites 
`site-003` and `site-012` PRAXIS datasets.

The following models were trained:
1. REGION-BASED (a single input channel (T2w_sag) to segment both SC and lesions):
   1. The default nnUNet trainer `nnUNetTrainer` (i.e., the sum of Cross Entropy Loss and Dice Loss **with** the smoothing term)
   2. `nnUNetTrainerDiceCELoss_noSmooth` (i.e., the sum of Cross Entropy Loss and Dice Loss **without** the smoothing term)
2. MULTI-CHANNEL (two input channels (T2w_sag and SC seg) as input channels to segment lesions):
   1. The default nnUNet trainer `nnUNetTrainer` (i.e., the sum of Cross Entropy Loss and Dice Loss **with** the smoothing term)
   2. `nnUNetTrainerDiceCELoss_noSmooth` (i.e., the sum of Cross Entropy Loss and Dice Loss **without** the smoothing term)