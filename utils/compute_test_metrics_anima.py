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

NOTE: Check out the latest version of ANIMA here: https://github.com/Inria-Empenn/Anima-Public/releases
and modify the folder names during the installation process accordingly. 
(at the time of writing, the latest version is 4.2, hence the folder names below)

INSTALLATION:
##### STEP 0: Install git lfs via apt if you don't already have it.
##### STEP 1: Install ANIMA #####
cd ~
mkdir anima_4.2/
cd anima_4.2/
wget -q https://github.com/Inria-Visages/Anima-Public/releases/download/v4.2/Anima-Ubuntu-4.2.zip   (change version to latest)
unzip Anima-Ubuntu-4.2.zip
git lfs install
git clone --depth 1 https://github.com/Inria-Visages/Anima-Scripts-Public.git
git clone --depth 1 https://github.com/Inria-Visages/Anima-Scripts-Data-Public.git
cd ~
mkdir .anima_4.2/
touch .anima_4.2/config.txt
nano .anima_4.2/config.txt

##### STEP 2: Configure directories #####
# Variable names and section titles should stay the same
# Put this file in your HomeFolder/.anima/config.txt
# Make the anima variable point to your Anima public build
# Make the extra-data-root point to the data folder of Anima-Scripts
# The last folder separator for each path is crucial, do not forget them
# Use full paths, nothing relative or using tildes 

[anima-scripts]
anima = /home/<your-user-name>/anima_4.2/Anima-Binaries-4.2/
anima-scripts-public-root = /home/<your-user-name>/anima_4.2/Anima-Scripts-Public/
extra-data-root = /home/<your-user-name>/anima_4.2/Anima-Scripts-Data-Public/

USAGE:
python compute_test_metrics_anima.py --pred_folder <path_to_predictions_folder> 
--gt_folder <path_to_gt_folder> -dname <dataset_name> -o <output_folder>


NOTE 1: For checking all the available options run the following command from your terminal: 
      <anima_binaries_path>/animaSegPerfAnalyzer -h
NOTE 2: We use certain additional arguments below with the following purposes:
      -i -> input image, -r -> reference image, -o -> output folder
      -d -> evaluates surface distance, -l -> evaluates the detection of lesions
      -a -> intra-lesion evalulation (advanced), -s -> segmentation evaluation, 
      -X -> save as XML file  -A -> prints details on output metrics and exits


"""

import os
import glob
import subprocess
import argparse
from collections import defaultdict
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import nibabel as nib

# get the ANIMA binaries path
cmd = r'''grep "^anima = " ~/.anima_4.2/config.txt | sed "s/.* = //"'''
anima_binaries_path = subprocess.check_output(cmd, shell=True).decode('utf-8').strip('\n')
print('ANIMA Binaries Path:', anima_binaries_path)
print('Running ANIMA version:', subprocess.check_output(anima_binaries_path + 'animaSegPerfAnalyzer --version', shell=True).decode('utf-8').strip('\n'))

# Define arguments
parser = argparse.ArgumentParser(description='Compute test metrics using animaSegPerfAnalyzer')

# Arguments for model, data, and training
parser.add_argument('--pred-folder', required=True, type=str,
                    help='Path to the folder containing nifti images of test predictions')
parser.add_argument('--gt-folder', required=True, type=str,
                    help='Path to the folder containing nifti images of GT labels')
# parser.add_argument('-dnum', '--dataset-number', required=True, type=str,
#                     help='Dataset ID/Number used for creating datasets for nnUNet')
parser.add_argument('-dname', '--dataset-name', required=True, type=str,
                    help='Dataset name used for creating datasets for nnUNet')                    
parser.add_argument('-o', '--output-folder', required=True, type=str,
                    help='Path to the output folder to save the test metrics results')

args = parser.parse_args()


def get_test_metrics_by_dataset(pred_folder, gt_folder, data_set):
# def get_test_metrics_by_dataset(data_set):
    """
    Computes the test metrics given folders containing nifti images of test predictions 
    and GT images by running the "animaSegPerfAnalyzer" command
    """

    if data_set == "zurich":

        # glob all the predictions and GTs and get the last three digits of the filename
        predictions = sorted(glob.glob(os.path.join(pred_folder, "*_sub-zh*.nii.gz")))
        predictions = [p.split('/')[-1].split('.')[0].split('_')[3] for p in predictions]
        gts = sorted(glob.glob(os.path.join(gt_folder, "*_sub-zh*.nii.gz")))
        gts = [g.split('/')[-1].split('.')[0].split('_')[3] for g in gts]

        # print(predictions, "\n", gts)
        # make sure the predictions and GTs are in the same order
        assert predictions == gts, 'Predictions and GTs are not in the same order. Please check the filenames.'

        # loop over the predictions and compute the metrics
        for idx in predictions:
            
            # Load the predictions and GTs
            pred_file = glob.glob(os.path.join(pred_folder, f"*_sub-zh*{idx}.nii.gz"))
            # get the subject and session IDs from the pred filename
            sub_ses_pred = pred_file[0].split('/')[-1].split('.')[0].split('_')[1] + "_" + pred_file[0].split('/')[-1].split('.')[0].split('_')[2]
            # load the predictions
            pred_npy = nib.load(pred_file[0]).get_fdata()
            # make sure the predictions are binary because ANIMA accepts binarized inputs only
            pred_npy = np.array(pred_npy > 0.5, dtype=float)

            gt_file = glob.glob(os.path.join(gt_folder, f"*_sub-zh*_{idx}.nii.gz"))
            # get the subject and session IDs from the gt filename
            sub_ses_gt = gt_file[0].split('/')[-1].split('.')[0].split('_')[1] + "_" + gt_file[0].split('/')[-1].split('.')[0].split('_')[2]
            # load the GTs
            gt_npy = nib.load(gt_file[0]).get_fdata()
            # make sure the GT is binary because ANIMA accepts binarized inputs only
            gt_npy = np.array(gt_npy > 0.5, dtype=float)
            # print(((gt_npy==0.0) | (gt_npy==1.0)).all())

            # Save the binarized predictions and GTs
            pred_nib = nib.Nifti1Image(pred_npy, affine=np.eye(4))
            gtc_nib = nib.Nifti1Image(gt_npy, affine=np.eye(4))
            # make sure the subject and session IDs match
            assert sub_ses_pred == sub_ses_gt, 'Subject and session IDs for Preds and GTs do not match. Please check the filenames.'
            nib.save(img=pred_nib, filename=os.path.join(pred_folder, f"{args.dataset_name}_{sub_ses_pred}_{idx}_bin.nii.gz"))
            nib.save(img=gtc_nib, filename=os.path.join(gt_folder, f"{args.dataset_name}_{sub_ses_gt}_{idx}_bin.nii.gz"))

            # Run ANIMA segmentation performance metrics on the predictions            
            seg_perf_analyzer_cmd = '%s -i %s -r %s -o %s -d -l -s -X'
            os.system(seg_perf_analyzer_cmd %
                        (os.path.join(anima_binaries_path, 'animaSegPerfAnalyzer'),
                        os.path.join(pred_folder, f"{args.dataset_name}_{sub_ses_pred}_{idx}_bin.nii.gz"),
                        os.path.join(gt_folder, f"{args.dataset_name}_{sub_ses_gt}_{idx}_bin.nii.gz"),
                        os.path.join(args.output_folder, f"{idx}")))


            # Delete temporary binarized NIfTI files
            os.remove(os.path.join(pred_folder, f"{args.dataset_name}_{sub_ses_pred}_{idx}_bin.nii.gz"))
            os.remove(os.path.join(gt_folder, f"{args.dataset_name}_{sub_ses_gt}_{idx}_bin.nii.gz"))

        # Get all XML filepaths where ANIMA performance metrics are saved for each hold-out subject
        subject_filepaths = [os.path.join(args.output_folder, f) for f in
                                os.listdir(args.output_folder) if f.endswith('.xml')]
        
        return subject_filepaths

    elif data_set == "zurich-region":

        # glob all the predictions and GTs and get the last three digits of the filename
        predictions = sorted(glob.glob(os.path.join(pred_folder, "*_sub-zh*.nii.gz")))
        predictions = [p.split('/')[-1].split('.')[0].split('_')[3] for p in predictions]
        gts = sorted(glob.glob(os.path.join(gt_folder, "*_sub-zh*.nii.gz")))
        gts = [g.split('/')[-1].split('.')[0].split('_')[3] for g in gts]

        # print(predictions, "\n", gts)
        # make sure the predictions and GTs are in the same order
        assert predictions == gts, 'Predictions and GTs are not in the same order. Please check the filenames.'

        # loop over the predictions and compute the metrics
        for idx in predictions:
            
            # Load the predictions and GTs
            pred_file = glob.glob(os.path.join(pred_folder, f"*_sub-zh*{idx}.nii.gz"))
            # get the subject and session IDs from the pred filename
            sub_ses_pred = pred_file[0].split('/')[-1].split('.')[0].split('_')[1] + "_" + pred_file[0].split('/')[-1].split('.')[0].split('_')[2]

            gt_file = glob.glob(os.path.join(gt_folder, f"*_sub-zh*_{idx}.nii.gz"))
            # get the subject and session IDs from the gt filename
            sub_ses_gt = gt_file[0].split('/')[-1].split('.')[0].split('_')[1] + "_" + gt_file[0].split('/')[-1].split('.')[0].split('_')[2]

            for seg in ['sc', 'lesion']:
                
                # load the predictions and GTs
                pred_npy = nib.load(pred_file[0]).get_fdata()
                gt_npy = nib.load(gt_file[0]).get_fdata()

                if seg == 'sc':
                    pred_npy = np.array(pred_npy == 1, dtype=float)
                    gt_npy = np.array(gt_npy == 1, dtype=float)                
                
                elif seg == 'lesion':
                    pred_npy = np.array(pred_npy == 2, dtype=float)
                    gt_npy = np.array(gt_npy == 2, dtype=float)
                
                # Save the binarized predictions and GTs
                pred_nib = nib.Nifti1Image(pred_npy, affine=np.eye(4))
                gtc_nib = nib.Nifti1Image(gt_npy, affine=np.eye(4))
                # make sure the subject and session IDs match
                assert sub_ses_pred == sub_ses_gt, 'Subject and session IDs for Preds and GTs do not match. Please check the filenames.'
                nib.save(img=pred_nib, filename=os.path.join(pred_folder, f"{args.dataset_name}_{sub_ses_pred}_{idx}_{seg}.nii.gz"))
                nib.save(img=gtc_nib, filename=os.path.join(gt_folder, f"{args.dataset_name}_{sub_ses_gt}_{idx}_{seg}.nii.gz"))

                # Run ANIMA segmentation performance metrics on the predictions
                if seg == 'sc':
                    seg_perf_analyzer_cmd = '%s -i %s -r %s -o %s -d -s -X'
                elif seg == 'lesion':
                    seg_perf_analyzer_cmd = '%s -i %s -r %s -o %s -d -s -l -X'
                
                os.system(seg_perf_analyzer_cmd %
                            (os.path.join(anima_binaries_path, 'animaSegPerfAnalyzer'),
                            os.path.join(pred_folder, f"{args.dataset_name}_{sub_ses_pred}_{idx}_{seg}.nii.gz"),
                            os.path.join(gt_folder, f"{args.dataset_name}_{sub_ses_gt}_{idx}_{seg}.nii.gz"),
                            os.path.join(args.output_folder, f"{idx}_{seg}")))

                # Delete temporary binarized NIfTI files
                os.remove(os.path.join(pred_folder, f"{args.dataset_name}_{sub_ses_pred}_{idx}_{seg}.nii.gz"))
                os.remove(os.path.join(gt_folder, f"{args.dataset_name}_{sub_ses_gt}_{idx}_{seg}.nii.gz"))


        # Get all XML filepaths where ANIMA performance metrics are saved for each hold-out subject
        subject_sc_filepaths = [os.path.join(args.output_folder, f) for f in
                                os.listdir(args.output_folder) if f.endswith('.xml') and 'sc' in f]
        subject_lesion_filepaths = [os.path.join(args.output_folder, f) for f in
                                os.listdir(args.output_folder) if f.endswith('.xml') and 'lesion' in f]
        
        return subject_sc_filepaths, subject_lesion_filepaths


    elif data_set in ["spine-generic", "dcm-zurich", "sci-colorado"]:
        
        # glob all the predictions and GTs and get the last three digits of the filename
        predictions = sorted(glob.glob(os.path.join(pred_folder, "*.nii.gz")))
        predictions = [p.split('/')[-1].split('.')[0].split('_')[-1] for p in predictions]
        gts = sorted(glob.glob(os.path.join(gt_folder, "*.nii.gz")))
        gts = [g.split('/')[-1].split('.')[0].split('_')[-1] for g in gts]

        print(predictions, "\n", gts)

        # make sure the predictions and GTs are in the same order
        assert predictions == gts, 'Predictions and GTs are not in the same order. Please check the filenames.'

        # loop over the predictions and compute the metrics
        for idx in predictions:
            
            # Load the predictions and GTs
            # pred_file = os.path.join(pred_folder, f"{args.dataset_name}_*{(idx+1):03d}.nii.gz")
            pred_file = glob.glob(os.path.join(pred_folder, f"{args.dataset_name}_*{idx}.nii.gz"))
            # get the subject and session IDs from the pred filename
            # sub_ses_pred = pred_file[0].split('/')[-1].split('.')[0].split('_')[1] + "_" + pred_file[0].split('/')[-1].split('.')[0].split('_')[2]
            # load the predictions
            pred_npy = nib.load(pred_file[0]).get_fdata()
            # make sure the predictions are binary because ANIMA accepts binarized inputs only
            pred_npy = np.array(pred_npy > 0.5, dtype=float)

            # gt_file = os.path.join(gt_folder, f"{args.dataset_name}_{(idx+1):03d}.nii.gz")
            gt_file = glob.glob(os.path.join(gt_folder, f"{args.dataset_name}_*{idx}.nii.gz"))
            # get the subject and session IDs from the gt filename
            # sub_ses_gt = gt_file[0].split('/')[-1].split('.')[0].split('_')[1] + "_" + gt_file[0].split('/')[-1].split('.')[0].split('_')[2]
            # load the GTs
            gt_npy = nib.load(gt_file[0]).get_fdata()
            # make sure the GT is binary because ANIMA accepts binarized inputs only
            gt_npy = np.array(gt_npy > 0.5, dtype=float)
            # print(((gt_npy==0.0) | (gt_npy==1.0)).all())

            # Save the binarized predictions and GTs
            pred_nib = nib.Nifti1Image(pred_npy, affine=np.eye(4))
            gtc_nib = nib.Nifti1Image(gt_npy, affine=np.eye(4))
            # make sure the subject and session IDs match
            # assert sub_ses_pred == sub_ses_gt, 'Subject and session IDs for Preds and GTs do not match. Please check the filenames.'
            # nib.save(img=pred_nib, filename=os.path.join(pred_folder, f"{args.dataset_name}_{sub_ses_pred}_{(idx+1):03d}_bin.nii.gz"))
            # nib.save(img=gtc_nib, filename=os.path.join(gt_folder, f"{args.dataset_name}_{sub_ses_gt}_{(idx+1):03d}_bin.nii.gz"))
            nib.save(img=pred_nib, filename=os.path.join(pred_folder, f"{args.dataset_name}_{idx}_bin.nii.gz"))
            nib.save(img=gtc_nib, filename=os.path.join(gt_folder, f"{args.dataset_name}_{idx}_bin.nii.gz"))

            # Run ANIMA segmentation performance metrics on the predictions            
            seg_perf_analyzer_cmd = '%s -i %s -r %s -o %s -d -s -X'
            # os.system(seg_perf_analyzer_cmd %
            #             (os.path.join(anima_binaries_path, 'animaSegPerfAnalyzer'),
            #             os.path.join(pred_folder, f"{args.dataset_name}_{sub_ses_pred}_{(idx+1):03d}_bin.nii.gz"),
            #             os.path.join(gt_folder, f"{args.dataset_name}_{sub_ses_gt}_{(idx+1):03d}_bin.nii.gz"),
            #             os.path.join(args.output_folder, f"{(idx+1)}")))
            os.system(seg_perf_analyzer_cmd %
                        (os.path.join(anima_binaries_path, 'animaSegPerfAnalyzer'),
                        os.path.join(pred_folder, f"{args.dataset_name}_{idx}_bin.nii.gz"),
                        os.path.join(gt_folder, f"{args.dataset_name}_{idx}_bin.nii.gz"),
                        os.path.join(args.output_folder, f"{idx}")))


            # Delete temporary binarized NIfTI files
            # os.remove(os.path.join(pred_folder, f"{args.dataset_name}_{sub_ses_pred}_{(idx+1):03d}_bin.nii.gz"))
            # os.remove(os.path.join(gt_folder, f"{args.dataset_name}_{sub_ses_gt}_{(idx+1):03d}_bin.nii.gz"))
            os.remove(os.path.join(pred_folder, f"{args.dataset_name}_{idx}_bin.nii.gz"))
            os.remove(os.path.join(gt_folder, f"{args.dataset_name}_{idx}_bin.nii.gz"))


        # Get all XML filepaths where ANIMA performance metrics are saved for each hold-out subject
        subject_filepaths = [os.path.join(args.output_folder, f) for f in
                                os.listdir(args.output_folder) if f.endswith('.xml')]
        
        return subject_filepaths


# define variables
pred_folder, gt_folder = args.pred_folder, args.gt_folder
# dataset = "spine-generic"  # args.dataset  # TODO: create argument parser for this
dataset = "zurich"  # args.dataset  # TODO: create argument parser for this
# dataset = "zurich-region"  # args.dataset  # TODO: create argument parser for this
# dataset = "dcm-zurich"  # args.dataset  # TODO: create argument parser for this
# dataset = "sci-colorado"  # args.dataset  # TODO: create argument parser for this

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder, exist_ok=True)


if dataset != "zurich-region":

    # Get all XML filepaths where ANIMA performance metrics are saved for each hold-out subject
    subject_filepaths = get_test_metrics_by_dataset(pred_folder, gt_folder, data_set=dataset)

    test_metrics = defaultdict(list)

    # Update the test metrics dictionary by iterating over all subjects
    for subject_filepath in subject_filepaths:
        # print(subject_filepath)
        subject = os.path.split(subject_filepath)[-1].split('_')[0]
        root_node = ET.parse(source=subject_filepath).getroot()

        # if GT is empty then metrics aren't calculated, hence the only entries in the XML file 
        # NbTestedLesions and VolTestedLesions, both of which are zero. Hence, we can skip subjects
        # with empty GTs by checked if the length of the .xml file is 2
        if len(root_node) == 2:
            print(f"Skipping Subject={int(subject):03d} ENTIRELY Due to Empty GT!")
            continue

        # # Check if RelativeVolumeError is INF -> means the GT is empty and should be ignored
        # rve_metric = list(root_node)[6]
        # assert rve_metric.get('name') == 'RelativeVolumeError'
        # if np.isinf(float(rve_metric.text)):
        #     print('Skipping Subject=%s ENTIRELY Due to Empty GT!' % subject)
        #     continue

        for metric in list(root_node):
            name, value = metric.get('name'), float(metric.text)

            if np.isinf(value) or np.isnan(value):
                print(f'Skipping Metric={name} for Subject={int(subject):03d} Due to INF or NaNs!')
                continue

            test_metrics[name].append(value)


    # Print aggregation of each metric via mean and standard dev.
    with open(os.path.join(args.output_folder, f'log_{dataset}.txt'), 'a') as f:
        print('Test Phase Metrics [ANIMA]: ', file=f)

    print('Test Phase Metrics [ANIMA]: ')
    for key in test_metrics:
        print('\t%s -> Mean: %0.4f Std: %0.2f' % (key, np.mean(test_metrics[key]), np.std(test_metrics[key])))
        
        # save the metrics to a log file
        with open(os.path.join(args.output_folder, f'log_{dataset}.txt'), 'a') as f:
                    print("\t%s --> Mean: %0.3f, Std: %0.3f" % 
                            (key, np.mean(test_metrics[key]), np.std(test_metrics[key])), file=f)
    
else:

    # Get all XML filepaths where ANIMA performance metrics are saved for each hold-out subject
    subject_sc_filepaths, subject_lesion_filepaths = get_test_metrics_by_dataset(pred_folder, gt_folder, data_set=dataset)

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
        with open(os.path.join(args.output_folder, f'log_{dataset}.txt'), 'a') as f:
            print(f'Test Phase Metrics [ANIMA] for {seg_type}: ', file=f)

        print(f'Test Phase Metrics [ANIMA] for {seg_type}: ')
        for key in test_metrics:
            print('\t%s -> Mean: %0.4f Std: %0.2f' % (key, np.mean(test_metrics[key]), np.std(test_metrics[key])))
            
            # save the metrics to a log file
            with open(os.path.join(args.output_folder, f'log_{dataset}.txt'), 'a') as f:
                        print("\t%s --> Mean: %0.3f, Std: %0.3f" % 
                                (key, np.mean(test_metrics[key]), np.std(test_metrics[key])), file=f)