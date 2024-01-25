"""
This script evaluates the reference segmentations and model predictions 
using the "animaSegPerfAnalyzer" command

****************************************************************************************
SegPerfAnalyser (Segmentation Performance Analyzer) provides different marks, metrics 
and scores for segmentation evaluation.
3 categories are available:
    - SEGMENTATION EVALUATION:
        Dice, the mean overlap
        Jaccard, the union overlap
        Sensitivity
        Specificity
        NPV (Negative Predictive Value)
        PPV (Positive Predictive Value)
        RVE (Relative Volume Error) in percentage
    - SURFACE DISTANCE EVALUATION:
        Hausdorff distance
        Contour mean distance
        Average surface distance
    - DETECTION LESIONS EVALUATION:
        PPVL (Positive Predictive Value for Lesions)
        SensL, Lesion detection sensitivity
        F1 Score, a F1 Score between PPVL and SensL

Results are provided as follows: 
Jaccard;    Dice;   Sensitivity;    Specificity;    PPV;    NPV;    RelativeVolumeError;    
HausdorffDistance;  ContourMeanDistance;    SurfaceDistance;  PPVL;   SensL;  F1_score;       

NbTestedLesions;    VolTestedLesions;  --> These metrics are computed for images that 
                                            have no lesions in the GT
****************************************************************************************

Mathematical details on how these metrics are computed can be found here:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6135867/pdf/41598_2018_Article_31911.pdf

and in Section 4 of this paper (for how the subjects with no lesions are handled):
https://portal.fli-iam.irisa.fr/files/2021/06/MS_Challenge_Evaluation_Challengers.pdf

INSTALLATION:
##### STEP 0: Install git lfs via apt if you don't already have it.
##### STEP 1: Install ANIMA #####
cd ~
mkdir anima/
cd anima/
wget -q https://github.com/Inria-Visages/Anima-Public/releases/download/v4.2/Anima-Ubuntu-4.2.zip   (change version to latest)
unzip Anima-Ubuntu-4.2.zip
git lfs install
git clone --depth 1 https://github.com/Inria-Visages/Anima-Scripts-Public.git
git clone --depth 1 https://github.com/Inria-Visages/Anima-Scripts-Data-Public.git

##### STEP 2: Configure directories #####
# Variable names and section titles should stay the same
# Put this file in ${HOME}/.anima/config.txt
# Make the anima variable point to your Anima public build
# Make the extra-data-root point to the data folder of Anima-Scripts
# The last folder separator for each path is crucial, do not forget them
# Use full paths, nothing relative or using tildes

cd ~
mkdir .anima/
touch .anima/config.txt

echo "[anima-scripts]" >> .anima/config.txt
echo "anima = ${HOME}/anima/Anima-Binaries-4.2/" >> .anima/config.txt
echo "anima-scripts-public-root = ${HOME}/anima/Anima-Scripts-Public/" >> .anima/config.txt
echo "extra-data-root = ${HOME}/anima/Anima-Scripts-Data-Public/" >> .anima/config.txt

USAGE:
python compute_anima_metrics.py --pred_folder <path_to_predictions_folder> 
--gt_folder <path_to_gt_folder> -dname <dataset_name> --label-type <sc/lesion>


NOTE 1: For checking all the available options run the following command from your terminal: 
      <anima_binaries_path>/animaSegPerfAnalyzer -h
NOTE 2: We use certain additional arguments below with the following purposes:
      -i -> input image, -r -> reference image, -o -> output folder
      -d -> evaluates surface distance, -l -> evaluates the detection of lesions
      -a -> intra-lesion evalulation (advanced), -s -> segmentation evaluation, 
      -X -> save as XML file  -A -> prints details on output metrics and exits

Authors: Naga Karthik, Jan Valosek
"""

import os
import glob
import subprocess
import argparse
from collections import defaultdict
import xml.etree.ElementTree as ET
import numpy as np
import nibabel as nib
from test_utils import fetch_filename_details


REGION_BASED_DATASETS = ["site_003", "site_012"]
STANDARD_DATASETS = ["spine-generic", "sci-colorado", "sci-zurich", "basel-mp2rage"]


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Compute test metrics using animaSegPerfAnalyzer')

    # Arguments for model, data, and training
    parser.add_argument('--pred-folder', required=True, type=str,
                        help='Path to the folder containing nifti images of test predictions')
    parser.add_argument('--gt-folder', required=True, type=str,
                        help='Path to the folder containing nifti images of GT labels')
    parser.add_argument('-dname', '--dataset-name', required=True, type=str,
                        help='Dataset name used for storing on git-annex. For region-based metrics, '
                             'append "-region" to the dataset name')
    parser.add_argument('--label-type', required=True, type=str, choices=['sc', 'lesion'],
                        help='Type of prediction and GT label to be used for ANIMA evaluation.'
                            'Options: "sc" for spinal cord segmentation, "lesion" for lesion segmentation'
                            'NOTE: when label-type is "lesion", additional lesion detection metrics, namely,'
                            'Lesion PPV, Lesion Sensitivity, and F1_score are computed')
    # parser.add_argument('-o', '--output-folder', required=True, type=str,
    #                     help='Path to the output folder to save the test metrics results')

    return parser


def get_test_metrics_by_dataset(pred_folder, gt_folder, output_folder, anima_binaries_path, data_set, label_type):
    """
    Computes the test metrics given folders containing nifti images of test predictions 
    and GT images by running the "animaSegPerfAnalyzer" command
    """
    
    if data_set in REGION_BASED_DATASETS:

        # glob all the predictions and GTs and get the last three digits of the filename
        pred_files = sorted(glob.glob(os.path.join(pred_folder, "*.nii.gz")))
        gt_files = sorted(glob.glob(os.path.join(gt_folder, "*.nii.gz")))

        dataset_name_nnunet = fetch_filename_details(pred_files[0])[0]

        # loop over the predictions and compute the metrics
        for pred_file, gt_file in zip(pred_files, gt_files):
            
            _, sub_pred, ses_pred, idx_pred, _ = fetch_filename_details(pred_file)
            _, sub_gt, ses_gt, idx_gt, _ = fetch_filename_details(gt_file)

            # make sure the subject and session IDs match
            print(f"Subject and session IDs for Preds and GTs: {sub_pred}_{ses_pred}_{idx_pred}, "
                  f"{sub_gt}_{ses_gt}_{idx_gt}")
            assert idx_pred == idx_gt, ('Subject and session IDs for Preds and GTs do not match. '
                                        'Please check the filenames.')
            
            if ses_gt == "":
                sub_ses_pred, sub_ses_gt = f"{sub_pred}", f"{sub_gt}"
            else:
                sub_ses_pred, sub_ses_gt = f"{sub_pred}_{ses_pred}", f"{sub_gt}_{ses_gt}"
            assert sub_ses_pred == sub_ses_gt, ('Subject and session IDs for Preds and GTs do not match. '
                                                'Please check the filenames.')

            for seg in ['sc', 'lesion']:
                # load the predictions and GTs
                pred_npy = nib.load(pred_file).get_fdata()
                gt_npy = nib.load(gt_file).get_fdata()

                if seg == 'sc':
                    pred_npy = np.array(pred_npy == 1, dtype=float)
                    gt_npy = np.array(gt_npy == 1, dtype=float)                
                
                elif seg == 'lesion':
                    pred_npy = np.array(pred_npy == 2, dtype=float)
                    gt_npy = np.array(gt_npy == 2, dtype=float)
                
                # Save the binarized predictions and GTs
                pred_nib = nib.Nifti1Image(pred_npy, affine=np.eye(4))
                gtc_nib = nib.Nifti1Image(gt_npy, affine=np.eye(4))
                nib.save(img=pred_nib, filename=os.path.join(pred_folder, f"{dataset_name_nnunet}_{sub_ses_pred}_{idx_pred}_{seg}.nii.gz"))
                nib.save(img=gtc_nib, filename=os.path.join(gt_folder, f"{dataset_name_nnunet}_{sub_ses_gt}_{idx_gt}_{seg}.nii.gz"))

                # Run ANIMA segmentation performance metrics on the predictions
                if seg == 'sc':
                    seg_perf_analyzer_cmd = '%s -i %s -r %s -o %s -d -s -X'
                elif seg == 'lesion':   # add lesion evaluation metrics with `-l`
                    seg_perf_analyzer_cmd = '%s -i %s -r %s -o %s -d -s -l -X'
                
                os.system(seg_perf_analyzer_cmd %
                            (os.path.join(anima_binaries_path, 'animaSegPerfAnalyzer'),
                            os.path.join(pred_folder, f"{dataset_name_nnunet}_{sub_ses_pred}_{idx_pred}_{seg}.nii.gz"),
                            os.path.join(gt_folder, f"{dataset_name_nnunet}_{sub_ses_gt}_{idx_gt}_{seg}.nii.gz"),
                            os.path.join(output_folder, f"{idx_pred}_{seg}")))

                # Delete temporary binarized NIfTI files
                os.remove(os.path.join(pred_folder, f"{dataset_name_nnunet}_{sub_ses_pred}_{idx_pred}_{seg}.nii.gz"))
                os.remove(os.path.join(gt_folder, f"{dataset_name_nnunet}_{sub_ses_gt}_{idx_gt}_{seg}.nii.gz"))

        # Get all XML filepaths where ANIMA performance metrics are saved for each hold-out subject
        subject_sc_filepaths = [os.path.join(output_folder, f) for f in
                                os.listdir(output_folder) if f.endswith('.xml') and 'sc' in f]
        subject_lesion_filepaths = [os.path.join(output_folder, f) for f in
                                os.listdir(output_folder) if f.endswith('.xml') and 'lesion' in f]
        
        return subject_sc_filepaths, subject_lesion_filepaths

    elif data_set in STANDARD_DATASETS:
        # glob all the predictions and GTs and get the last three digits of the filename
        pred_files = sorted(glob.glob(os.path.join(pred_folder, "*.nii.gz")))
        gt_files = sorted(glob.glob(os.path.join(gt_folder, "*.nii.gz")))

        dataset_name_nnunet = fetch_filename_details(pred_files[0])[0]

        # loop over the predictions and compute the metrics
        for pred_file, gt_file in zip(pred_files, gt_files):
            
            _, sub_pred, ses_pred, idx_pred, _ = fetch_filename_details(pred_file)
            _, sub_gt, ses_gt, idx_gt, _ = fetch_filename_details(gt_file)

            # make sure the subject and session IDs match
            print(f"Subject and session IDs for Preds and GTs: {sub_pred}_{ses_pred}_{idx_pred}, {sub_gt}_{ses_gt}_{idx_gt}")
            assert idx_pred == idx_gt, 'Subject and session IDs for Preds and GTs do not match. Please check the filenames.'
            
            if ses_gt == "":
                sub_ses_pred, sub_ses_gt = f"{sub_pred}", f"{sub_gt}"
            else:
                sub_ses_pred, sub_ses_gt = f"{sub_pred}_{ses_pred}", f"{sub_gt}_{ses_gt}"
            assert sub_ses_pred == sub_ses_gt, 'Subject and session IDs for Preds and GTs do not match. Please check the filenames.'

            # load the predictions and GTs
            pred_npy = nib.load(pred_file).get_fdata()
            gt_npy = nib.load(gt_file).get_fdata()
            
            # make sure the predictions are binary because ANIMA accepts binarized inputs only
            pred_npy = np.array(pred_npy > 0.5, dtype=float)
            gt_npy = np.array(gt_npy > 0.5, dtype=float)

            # Save the binarized predictions and GTs
            pred_nib = nib.Nifti1Image(pred_npy, affine=np.eye(4))
            gtc_nib = nib.Nifti1Image(gt_npy, affine=np.eye(4))
            nib.save(img=pred_nib, filename=os.path.join(pred_folder, f"{dataset_name_nnunet}_{idx_pred}_bin.nii.gz"))
            nib.save(img=gtc_nib, filename=os.path.join(gt_folder, f"{dataset_name_nnunet}_{idx_gt}_bin.nii.gz"))

            # Run ANIMA segmentation performance metrics on the predictions            
            if label_type == 'lesion':
                 seg_perf_analyzer_cmd = '%s -i %s -r %s -o %s -d -l -s -X'
            elif label_type == 'sc':
                seg_perf_analyzer_cmd = '%s -i %s -r %s -o %s -d -s -X'
            else:
                raise ValueError('Please specify a valid label type: lesion or sc')

            os.system(seg_perf_analyzer_cmd %
                        (os.path.join(anima_binaries_path, 'animaSegPerfAnalyzer'),
                        os.path.join(pred_folder, f"{dataset_name_nnunet}_{idx_pred}_bin.nii.gz"),
                        os.path.join(gt_folder, f"{dataset_name_nnunet}_{idx_gt}_bin.nii.gz"),
                        os.path.join(output_folder, f"{idx_pred}")))

            # Delete temporary binarized NIfTI files
            os.remove(os.path.join(pred_folder, f"{dataset_name_nnunet}_{idx_pred}_bin.nii.gz"))
            os.remove(os.path.join(gt_folder, f"{dataset_name_nnunet}_{idx_gt}_bin.nii.gz"))

        # Get all XML filepaths where ANIMA performance metrics are saved for each hold-out subject
        subject_filepaths = [os.path.join(output_folder, f) for f in
                                os.listdir(output_folder) if f.endswith('.xml')]
        
        return subject_filepaths


def main():

    # get the ANIMA binaries path
    cmd = r'''grep "^anima = " ~/.anima/config.txt | sed "s/.* = //"'''
    anima_binaries_path = subprocess.check_output(cmd, shell=True).decode('utf-8').strip('\n')
    print('ANIMA Binaries Path:', anima_binaries_path)
    # version = subprocess.check_output(anima_binaries_path + 'animaSegPerfAnalyzer --version', shell=True).decode('utf-8').strip('\n')
    print('Running ANIMA version:',
          subprocess.check_output(anima_binaries_path + 'animaSegPerfAnalyzer --version', shell=True).decode(
              'utf-8').strip('\n'))

    parser = get_parser()
    args = parser.parse_args()

    # define variables
    pred_folder, gt_folder = args.pred_folder, args.gt_folder
    dataset_name = args.dataset_name
    label_type = args.label_type

    output_folder = os.path.join(pred_folder, f"anima_stats")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    print(f"Saving ANIMA performance metrics to {output_folder}")

    if dataset_name not in REGION_BASED_DATASETS:

        # Get all XML filepaths where ANIMA performance metrics are saved for each hold-out subject
        subject_filepaths = get_test_metrics_by_dataset(pred_folder, gt_folder, output_folder, anima_binaries_path,
                                                        data_set=dataset_name, label_type=label_type)

        test_metrics = defaultdict(list)

        # Update the test metrics dictionary by iterating over all subjects
        for subject_filepath in subject_filepaths:
            subject = os.path.split(subject_filepath)[-1].split('_')[0]
            root_node = ET.parse(source=subject_filepath).getroot()

            # if GT is empty then metrics aren't calculated, hence the only entries in the XML file 
            # NbTestedLesions and VolTestedLesions, both of which are zero. Hence, we can skip subjects
            # with empty GTs by checked if the length of the .xml file is 2
            if len(root_node) == 2:
                print(f"Skipping Subject={int(subject):03d} ENTIRELY Due to Empty GT!")
                continue

            for metric in list(root_node):
                name, value = metric.get('name'), float(metric.text)

                if np.isinf(value) or np.isnan(value):
                    print(f'Skipping Metric={name} for Subject={int(subject):03d} Due to INF or NaNs!')
                    continue

                test_metrics[name].append(value)

        # Print aggregation of each metric via mean and standard dev.
        with open(os.path.join(output_folder, f'log_{dataset_name}.txt'), 'a') as f:
            print('Test Phase Metrics [ANIMA]: ', file=f)

        print('Test Phase Metrics [ANIMA]: ')
        for key in test_metrics:
            print('\t%s -> Mean: %0.4f Std: %0.2f' % (key, np.mean(test_metrics[key]), np.std(test_metrics[key])))
            
            # save the metrics to a log file
            with open(os.path.join(output_folder, f'log_{dataset_name}.txt'), 'a') as f:
                        print("\t%s --> Mean: %0.3f, Std: %0.3f" % 
                                (key, np.mean(test_metrics[key]), np.std(test_metrics[key])), file=f)
        
    else:

        # Get all XML filepaths where ANIMA performance metrics are saved for each hold-out subject
        subject_sc_filepaths, subject_lesion_filepaths = \
            get_test_metrics_by_dataset(pred_folder, gt_folder, output_folder, anima_binaries_path,
                                        data_set=dataset_name, label_type=label_type)

        # loop through the sc and lesion filepaths and get the metrics
        for subject_filepaths in [subject_sc_filepaths, subject_lesion_filepaths]:
        
            test_metrics = defaultdict(list)

            # Update the test metrics dictionary by iterating over all subjects
            for subject_filepath in subject_filepaths:
                
                subject = os.path.split(subject_filepath)[-1].split('_')[0]
                seg_type = os.path.split(subject_filepath)[-1].split('_')[1]
                root_node = ET.parse(source=subject_filepath).getroot()

                # if GT is empty then metrics aren't calculated, hence the only entries in the XML file 
                # NbTestedLesions and VolTestedLesions, both of which are zero. Hence, we can skip subjects
                # with empty GTs by checked if the length of the .xml file is 2
                if len(root_node) == 2:
                    print(f"Skipping Subject={int(subject):03d} ENTIRELY Due to Empty GT!")
                    continue

                for metric in list(root_node):
                    name, value = metric.get('name'), float(metric.text)

                    if np.isinf(value) or np.isnan(value):
                        print(f'Skipping Metric={name} for Subject={int(subject):03d} Due to INF or NaNs!')
                        continue

                    test_metrics[name].append(value)

            # Print aggregation of each metric via mean and standard dev.
            with open(os.path.join(output_folder, f'log_{dataset_name}.txt'), 'a') as f:
                print(f'Test Phase Metrics [ANIMA] for {seg_type}: ', file=f)

            print(f'Test Phase Metrics [ANIMA] for {seg_type}: ')
            for key in test_metrics:
                print('\t%s -> Mean: %0.4f Std: %0.2f' % (key, np.mean(test_metrics[key]), np.std(test_metrics[key])))
                
                # save the metrics to a log file
                with open(os.path.join(output_folder, f'log_{dataset_name}.txt'), 'a') as f:
                            print("\t%s --> Mean: %0.3f, Std: %0.3f" % 
                                    (key, np.mean(test_metrics[key]), np.std(test_metrics[key])), file=f)


if __name__ == '__main__':
    main()
