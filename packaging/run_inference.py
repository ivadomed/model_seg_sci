import os
import argparse
import torch
import glob
import time
import numpy as np
import nibabel as nib
from packaging_utils import convert_filenames_to_nnunet_format, add_suffix, splitext, convert_to_rpi

from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data as predictor
# from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


"""
Usage example:
Method 1 (when running on whole dataset):
    python run_inference.py --path-dataset /path/to/test-dataset --path-out /path/to/output --path-model /path/to/model 
                                --pred-type lesion-seg

Method 2 (when running on individual images):
    python run_inference.py --path-images /path/to/image1 /path/to/image2 --path-out /path/to/output --path-model /path/to/model 
                                --pred-type lesion-seg                          
"""

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Segment images using nnUNet')
    parser.add_argument('--path-dataset', default=None, type=str,
                        help='Path to the test dataset folder. Use this argument only if you want '
                        'predict on a whole dataset.')
    parser.add_argument('--path-images', default=None, nargs='+', type=str,
                        help='List of images to segment. Use this argument only if you want '
                        'predict on a single image or list of invidiual images.')
    parser.add_argument('--path-out', help='Path to output directory.', required=True)
    parser.add_argument('--path-model', required=True, 
                        help='Path to the model directory. This folder should contain individual folders '
                        'like fold_0, fold_1, etc.',)
    parser.add_argument('--pred-type', default='lesion-seg', required=True,
                         choices=['sc-seg', 'lesion-seg', 'all'],
                        help='Type of prediction to obtain. If "all", then both spinal cord and lesion '
                        'segmentations will be obtained in the same nifti file. Default: lesion-seg')
    parser.add_argument('--use-gpu', action='store_true', default=False,
                        help='Use GPU for inference. Default: False')
    parser.add_argument('--use-best-checkpoint', action='store_true', default=False,
                        help='Use the best checkpoint (instead of the final checkpoint) for prediction. '
                        'NOTE: nnUNet by default uses the final checkpoint. Default: False')

    return parser


def main():

    parser = get_parser()
    args = parser.parse_args()

    if not os.path.exists(args.path_out):
        os.makedirs(args.path_out, exist_ok=True)

    if args.path_dataset is not None and args.path_images is not None:
        raise ValueError('You can only specify either --path-dataset or --path-images (not both). See --help for more info.')
    
    if args.path_dataset is not None:
        print('Found a dataset folder. Running inference on the whole dataset...')

        # NOTE: nnUNet only wants the _0000 suffix for files contained in a folder (i.e. when inference is run on a whole dataset)
        # hence, we create a temporary folder with the proper filenames and delete it after inference is done
        # More info about that naming convention here: 
        # https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format_inference.md
        
        print('Creating temporary folder with proper filenames...')
        path_data_tmp = convert_filenames_to_nnunet_format(args.path_dataset)
        path_out = args.path_out

        # convert all images to RPI orientation
        path_data_tmp, orig_orientation = convert_to_rpi(path_data_tmp)
        print(f'Original orientation: {orig_orientation}')

    elif args.path_images is not None:
        # NOTE: for individual images, the _0000 suffix is not needed. BUT, the images should be in a list of lists
        # get list of images from input argument
        print(f'Found {len(args.path_images)} images. Running inference on them...')
        # path_data_tmp = [[os.path.basename(f)] for f in args.path_images]
        path_data_tmp = [[f] for f in args.path_images]
        # print(path_data_tmp)

        # convert all images to RPI orientation
        path_data_tmp, orig_orientation = convert_to_rpi(path_data_tmp)
        print(f'Original orientation: {orig_orientation}')

        path_out = args.path_out
        # # add suffix '_pred' to predicted images
        # for f in args.path_images:
        #     path_pred = os.path.join(args.path_out, add_suffix(f, '_pred')) 
        #     path_out.append(path_pred)

    # uses all the folds available in the model folder by default
    folds_avail = [int(f.split('_')[-1]) for f in os.listdir(args.path_model) if f.startswith('fold_')]

    # ---------------------------------------------------------------
    # OPTION 1: Currently, pip install nnUNetv2 does not have the latest version of nnUNet's inference 
    # which is defined in OPTION 2. Hence, this method
    # ---------------------------------------------------------------

    print('Starting inference...')
    start = time.time()
    # directly call the predict function
    predictor(
        list_of_lists_or_source_folder=path_data_tmp, 
        output_folder=path_out,
        model_training_output_dir=args.path_model,
        use_folds=folds_avail,
        tile_step_size=0.5,
        use_gaussian=True,                                      # applies gaussian noise and gaussian blur
        use_mirroring=False,                                    # test time augmentation by mirroring on all axes
        perform_everything_on_gpu=True if args.use_gpu else False,
        device=torch.device('cuda', 0) if args.use_gpu else torch.device('cpu'),
        verbose=False,
        save_probabilities=False,
        overwrite=True,
        checkpoint_name='checkpoint_final.pth' if not args.use_best_checkpoint else 'checkpoint_best.pth',
        num_processes_preprocessing=3,
        num_processes_segmentation_export=3
    )
    end = time.time()
    
    # ---------------------------------------------------------------
    # OPTION 2
    # ---------------------------------------------------------------

    # instantiate the nnUNetPredictor
    # predictor = nnUNetPredictor(
    #     tile_step_size=0.5,
    #     use_gaussian=True,
    #     use_mirroring=True,
    #     perform_everything_on_gpu=True if args.use_gpu else False,
    #     device=torch.device('cuda', 0) if args.use_gpu else torch.device('cpu'),
    #     verbose=False,
    #     verbose_preprocessing=False,
    #     allow_tqdm=True
    # )
    # print('Running inference on device: {}'.format(predictor.device))

    # # initializes the network architecture, loads the checkpoint
    # predictor.initialize_from_trained_model_folder(
    #     join(args.path_model),
    #     use_folds=(0, 1),   # 
    #     checkpoint_name='checkpoint_final.pth',
    # )
    # print('Model loaded successfully. Fetching test data...')

    # # variant 1: give input and output folders
    # # adapted from: https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/inference
    # if args.path_data is not None:
    #     predictor.predict_from_files(path_data_tmp, path_out,
    #                                 save_probabilities=False, overwrite=False,
    #                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
    #                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    # # variant 2, use list of files as inputs. Note the usage of nested lists
    # if args.path_image is not None:
    #     # get absolute path to the image
    #     args.path_image = Path(args.path_image).absolute()

    #     predictor.predict_from_list_of_files([[args.path_image]], args.path_out,
    #                                          save_probabilities=False, overwrite=False,
    #                                          num_processes_preprocessing=2, num_processes_segmentation_export=2,
    #                                          folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    print('Inference done.')

    print('Deleting the temporary folder...')
    # delete the temporary folder
    os.system('rm -rf {}'.format(path_data_tmp))

    # reorient the images back to original orientation
    print('Re-orienting the predictions back to original orientation...')
    for f in os.listdir(path_out):
        if f.endswith('.nii.gz'):
            # get absolute path to the image
            f = os.path.join(path_out, f)

            # reorient the image to RPI using SCT
            os.system('sct_image -i {} -setorient {} -o {}'.format(f, orig_orientation, f))

    print(f'Reorientation to original orientation {orig_orientation} done.')

    # split the predictions into different sc-seg and lesion-seg
    if args.pred_type == 'all':
        # do nothing - keep the predictions as they are
        pass
    elif args.pred_type == 'sc-seg':
        out_folder = os.path.join(args.path_out, 'sc-seg')
        if not os.path.exists(out_folder):
            os.makedirs(out_folder, exist_ok=True)

        # get all the predictions
        pred_files = sorted(glob.glob(os.path.join(args.path_out, '*.nii.gz')))
        for pred in pred_files:
            # load the image
            img_nii = nib.load(pred)
            img = img_nii.get_fdata()

            # split the labels
            img_sc_seg = np.zeros_like(img)
            img_sc_seg[img == 1] = 1    # NOTE: this is only the SC without the lesion
            img_sc_seg[img == 2] = 1    # since lesion also has be to be included in the SC

            # save the images
            save_name = os.path.basename(pred).replace('.nii.gz', '_pred-sc.nii.gz')
            path_out = os.path.join(out_folder, save_name)
            nib.save(nib.Nifti1Image(img_sc_seg, img_nii.affine, img_nii.header), path_out)

    elif args.pred_type == 'lesion-seg':
        out_folder = os.path.join(args.path_out, 'lesion-seg')
        if not os.path.exists(out_folder):
            os.makedirs(out_folder, exist_ok=True)

        # get all the predictions
        pred_files = sorted(glob.glob(os.path.join(args.path_out, '*.nii.gz')))
        for pred in pred_files:
            # load the image
            img_nii = nib.load(pred)
            img = img_nii.get_fdata()

            # split the labels
            img_lesion_seg = np.zeros_like(img)
            img_lesion_seg[img == 2] = 1

            # save the images
            save_name = os.path.basename(pred).replace('.nii.gz', '_pred-lesion.nii.gz')
            path_out = os.path.join(out_folder, save_name)
            nib.save(nib.Nifti1Image(img_lesion_seg, img_nii.affine, img_nii.header), path_out)

    else:
        raise ValueError('Invalid value for --pred_type. Valid values are: [all, sc-seg, lesion-seg]')

    print('----------------------------------------------------')
    print('Results can be found in: {}'.format(args.path_out))
    print('----------------------------------------------------')

    total_time = end - start
    print('Total time elapsed: {} minute(s) {} seconds'.format(int(total_time // 60), int(round(total_time % 60))))
    print('----------------------------------------------------')

if __name__ == '__main__':
    main()