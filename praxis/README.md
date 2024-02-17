
## 01. Inference using the SCIseg nnUNet model

`01_run_inference_praxis.sh` was used to run inference on the `site-003` and `site-012` PRAXIS datasets. The script ran
our SCIseg nnUNet model to segment the spinal cord and SCI lesions. 

### Spinal cord

The predicted spinal cords were visually inspected and manually corrected if necessary (using `manual_correction.py`), 
added to the BIDS dataset under `derivatives/labels`, and pushed to the repository.

### SCI lesions

The lesions predicted by the SCIseg model were compared against the manual GT lesions (located under `derivatives/labels`) 
using ANIMA segmentation performance metrics. The SCIseg lesion segmentation showed low lesion segmentation performance.

## 02. Training pre-operative nnUNet model from scratch

`02_run_training_praxis.sh` was used to train the pre-operative SCIseg nnUNet model. The model was trained on sites 
`site-003` and `site-012` PRAXIS datasets.

Two models were trained:
1. The default nnUNet trainer `nnUNetTrainer` (i.e., the sum of Cross Entropy Loss and Dice Loss **with** the smoothing term)
2. `nnUNetTrainerDiceCELoss_noSmooth` (i.e., the sum of Cross Entropy Loss and Dice Loss **without** the smoothing term)