import numpy as np
import nibabel as nib 
import os
import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='Segment an image using nnUNet model.')
    parser.add_argument('--path-preds', help='Path to the folder containing the sct_run_batch results of all seeds', required=True)

    return parser

path_dcm_dataset = "/home/GRAMES.POLYMTL.CA/u114716/datasets/dcm-zurich-lesions"

def main():

    args = get_parser().parse_args()

    path_out = os.path.join(args.path_preds, "averaged_preds_across_seeds", "data_processed")
    if not os.path.exists(path_out):
        os.makedirs(path_out, exist_ok=True)

    subs_list = [subs for subs in os.listdir(path_dcm_dataset) if subs.startswith("sub-")]
    results_folders = [folder for folder in os.listdir(args.path_preds) if folder.startswith("seed")]

    for sub in subs_list:
        for label_type in ["sc", "lesion"]:
            preds_stack = []
            for fldr in results_folders:
                print(f"Processing subject {sub} for {label_type} from seed {fldr}")
                path_pred = os.path.join(args.path_preds, fldr, "data_processed", sub, "anat", f"{sub}_acq-ax_T2w_{label_type}_nnunet_3d_fullres.nii.gz")
                if not os.path.exists(path_pred):
                    print(f"WARNING: Prediction for {sub} does not exist... ! ")
                    break

                pred = nib.load(path_pred)
                preds_stack.append(pred.get_fdata())
        
            preds_stack = np.stack(preds_stack, axis=0)
        
            # Average predictions across seeds
            pred_avg = np.mean(preds_stack, axis=0)

            # binarize predictions using np.where
            pred_avg = np.where(pred_avg > 0.5, 1, 0)

            pred_avg_nii = nib.Nifti1Image(pred_avg, affine=pred.affine, header=pred.header)
            # call mkdir -p to create the folder if it doesn't exist
            save_path = os.path.join(path_out, sub, "anat")
            os.makedirs(save_path, exist_ok=True)
            nib.save(pred_avg_nii, os.path.join(save_path, f"{sub}_acq-ax_T2w_{label_type}_avg.nii.gz"))
            print("------------------------------------")

if __name__ == '__main__':
    main()