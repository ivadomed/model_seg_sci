import os
import json
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
import argparse
import nibabel as nib
from get_bbox_coords import get_bounding_boxes
from fold_generator import FoldGenerator


root = "/Users/nagakarthik/code/mri_datasets/sci-zurich_preprocessed_full/data_processed_clean"

parser = argparse.ArgumentParser(description='Code for creating k-fold splits of the ms-challenge-2021 dataset.')

parser.add_argument('-se', '--seed', default=42, type=int, help="Seed for reproducibility")
parser.add_argument('-ncv', '--num_cv_folds', default=5, type=int, help="To create a k-fold dataset for cross validation")
parser.add_argument('-dr', '--data_root', default=root, type=str, help='Path to the data set directory')

args = parser.parse_args()

root = args.data_root
# Get all subjects
subjects_df = pd.read_csv(os.path.join(root, 'participants.tsv'), sep='\t')
subjects = subjects_df['participant_id'].values.tolist()
print(len(subjects))

seed = args.seed
num_cv_folds = args.num_cv_folds    # for 100 subjects, performs a 60-20-20 split with num_cv_plots

# returns a nested list of length (num_cv_folds), each element (again, a list) consisting of 
# train, val, test indices and the fold number
names_list = FoldGenerator(seed, num_cv_folds, len_data=len(subjects)).get_fold_names()

for fold in range(num_cv_folds):

    train_ix, val_ix, test_ix, fold_num = names_list[fold]
    training_subjects = [subjects[tr_ix] for tr_ix in train_ix]
    validation_subjects = [subjects[v_ix] for v_ix in val_ix]
    test_subjects = [subjects[te_ix] for te_ix in test_ix]

    # keys to be defined in the dataset_0.json
    params = {}
    params["description"] = "sci-zurich naga"
    params["labels"] = {
        "0": "background",
        "1": "sc-lesion"
        }
    params["license"] = "nk"
    params["modality"] = {
        "0": "MRI"
        }
    params["name"] = "sci-zurich"
    params["numTest"] = len(test_subjects)
    params["numTraining"] = len(training_subjects) + len(validation_subjects)
    params["reference"] = "University of Zurich"
    params["tensorImageSize"] = "3D"


    train_val_subjects_dict = {
        "training": training_subjects,
        "validation": validation_subjects,
    } 
    test_subjects_dict =  {"test": test_subjects}

    # run loop for training and validation subjects
    for name, subs_list in train_val_subjects_dict.items():

        temp_list = []
        for subject_no, subject in enumerate(tqdm(subs_list, desc='Loading Volumes')):
        
            # Another for loop for going through sessions
            temp_subject_path = os.path.join(root, subject)
            num_sessions_per_subject = sum(os.path.isdir(os.path.join(temp_subject_path, pth)) for pth in os.listdir(temp_subject_path))

            for ses_idx in range(1, num_sessions_per_subject+1):
                temp_data = {}
                # Get paths with session numbers
                session = 'ses-0' + str(ses_idx)
                subject_images_path = os.path.join(root, subject, session, 'anat')
                subject_labels_path = os.path.join(root, 'derivatives', 'labels', subject, session, 'anat')

                subject_image_file = os.path.join(subject_images_path, '%s_%s_acq-sag_T2w.nii.gz' % (subject, session))
                subject_label_file = os.path.join(subject_labels_path, '%s_%s_acq-sag_T2w_lesion-manual.nii.gz' % (subject, session))

                # load GT mask
                gt_label = nib.load(subject_label_file).get_fdata()
                bbox_coords = get_bounding_boxes(mask=gt_label)

                # store in a temp dictionary
                temp_data["image"] = subject_image_file.replace(root+"/", '') # .strip(root)
                temp_data["label"] = subject_label_file.replace(root+"/", '') # .strip(root)
                temp_data["box"] = bbox_coords
                
                temp_list.append(temp_data)
        
        params[name] = temp_list

    # run separate loop for testing 
    for name, subs_list in test_subjects_dict.items():
        temp_list = []
        for subject_no, subject in enumerate(tqdm(subs_list, desc='Loading Volumes')):
        
            # Another for loop for going through sessions
            temp_subject_path = os.path.join(root, subject)
            num_sessions_per_subject = sum(os.path.isdir(os.path.join(temp_subject_path, pth)) for pth in os.listdir(temp_subject_path))

            for ses_idx in range(1, num_sessions_per_subject+1):
                temp_data = {}
                # Get paths with session numbers
                session = 'ses-0' + str(ses_idx)
                subject_images_path = os.path.join(root, subject, session, 'anat')
                subject_labels_path = os.path.join(root, 'derivatives', 'labels', subject, session, 'anat')
                
                subject_image_file = os.path.join(subject_images_path, '%s_%s_acq-sag_T2w.nii.gz' % (subject, session))
                subject_label_file = os.path.join(subject_labels_path, '%s_%s_acq-sag_T2w_lesion-manual.nii.gz' % (subject, session))

                # load GT mask
                gt_label = nib.load(subject_label_file).get_fdata()
                bbox_coords = get_bounding_boxes(mask=gt_label)

                temp_data["image"] = subject_image_file.replace(root+"/", '')
                temp_data["label"] = subject_label_file.replace(root+"/", '')
                temp_data["box"] = bbox_coords

                temp_list.append(temp_data)
        
        params[name] = temp_list

    final_json = json.dumps(params, indent=4, sort_keys=True)
    jsonFile = open(root + "/" + f"dataset_fold-{fold_num}.json", "w")
    jsonFile.write(final_json)
    jsonFile.close()




    



