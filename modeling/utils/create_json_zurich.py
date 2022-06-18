import os
import json
import pandas as pd
from tqdm import tqdm
import random
import numpy as np


root = "/Users/nagakarthik/code/mri_datasets/sci-zurich_preprocessed_full/data_processed_clean"

# Get all subjects
subjects_df = pd.read_csv(os.path.join(root, 'participants.tsv'), sep='\t')
subjects = subjects_df['participant_id'].values.tolist()
print(len(subjects))

seed = 42
random.seed(seed)
random.shuffle(subjects)

# define train/test split ratio
tr, va, te = 0.7, 0.1, 0.2

# split into train/test
test_subjects = subjects[:int(len(subjects) * te)]
train_val_subjects = subjects[int(len(subjects) * (1-tr-va)):]

# split into train/val
validation_subjects = train_val_subjects[:int(len(subjects) * va)]
training_subjects = train_val_subjects[int(len(subjects) * va):]

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
params["numTraining"] = len(train_val_subjects)
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
            
            # store in a temp dictionary
            temp_data["image"] = subject_image_file.replace(root+"/", '') # .strip(root)
            temp_data["label"] = subject_label_file.replace(root+"/", '') # .strip(root)

            temp_list.append(temp_data)
    
    params[name] = temp_list

# run separte loop for testing "without" the labels
for name, subs_list in test_subjects_dict.items():
    temp_list = []
    for subject_no, subject in enumerate(tqdm(subs_list, desc='Loading Volumes')):
    
        # Another for loop for going through sessions
        temp_subject_path = os.path.join(root, subject)
        num_sessions_per_subject = sum(os.path.isdir(os.path.join(temp_subject_path, pth)) for pth in os.listdir(temp_subject_path))

        for ses_idx in range(1, num_sessions_per_subject+1):
            # Get paths with session numbers
            session = 'ses-0' + str(ses_idx)
            subject_images_path = os.path.join(root, subject, session, 'anat')
            subject_image_file = os.path.join(subject_images_path, '%s_%s_acq-sag_T2w.nii.gz' % (subject, session))
            
            temp_list.append(subject_image_file.replace(root+"/", ''))
    
    params[name] = temp_list

final_json = json.dumps(params, indent=4, sort_keys=True)
jsonFile = open(root + "/" + "dataset_0.json", "w")
jsonFile.write(final_json)
jsonFile.close()




    



