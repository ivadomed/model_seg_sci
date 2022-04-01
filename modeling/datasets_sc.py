import os
import subprocess
from tqdm import tqdm
import random
from collections import defaultdict
import pandas as pd
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

import torch
from torch.utils.data import Dataset, random_split

from ivadomed.transforms import CenterCrop, RandomAffine, ElasticTransform, \
     NormalizeInstance, RandomReverse, RandomGamma, RandomBiasField, RandomBlur
from ivadomed.metrics import dice_score, hausdorff_score, intersection_over_union, \
    precision_score, recall_score, specificity_score, numeric_score
from utils import volume2subvolumes, subvolumes2volume


# ---------------------------- Datasets Implementation -----------------------------
class SCIZurichDataset(Dataset):
    """
    Custom PyTorch dataset for the Hyperintense T2 lesion segmentation in Acute SCI (Data from SCI-Zurich) 
    Works only with 3D subvolumes. Implements training, validation, and test phases. 
    Training and validation is utilized via the canonic __get_item__() function and by carefully setting 
    the `train` parameter before accessing an item  (e.g. via iterating over the dataloader in modeling/train.py). 
    Test phase is implemented with the `test` method.
    :param (float) fraction_data: Fraction of subjects to use for the entire dataset. Helps with debugging.
    :param (float) fraction_hold_out: Fraction of subjects to hold-out for the test phase. We want
           to hold-out entire patients, as opposed to subvolumes, to have a more representative
           test with the ANIMA script `animaSegPerfAnalyzer`.
    :param (tuple) center_crop_size: The 3D center-crop size for the volumes. For now, we can
           leave this at it's default value (320, 384, 512).
    :param (tuple) subvolume_size: The 3D subvolume size to be used in training & validation.
    :param (tuple) stride_size: The 3D stride size to be used in training & validation.
    :param (str) results_dir: The directory to where the save the results from test phase.
    :param (bool) visualize_test_preds: Set to True to save predictions as NIfTI files during test phase.
    :param (int) seed: Seed for reproducibility (e.g. we want the same train-val split with same seed)
    """
    def __init__(self, root, fraction_data=1.0, fraction_hold_out=0.2,
                 center_crop_size=(96, 384, 64), subvolume_size=(48, 64, 16),
                 stride_size=(16, 16, 16), only_eval=False, results_dir='outputs',
                 visualize_test_preds=True, seed=42):
        super(SCIZurichDataset).__init__()

        if only_eval and os.path.exists(results_dir):
            print('WARNING: results_dir=%s already exists! Files might be overwritten...' % results_dir)
        else:
            os.makedirs(results_dir)

        # # Set / create the results path for the test phase
        # if only_eval:
        #     if not os.path.exists(results_dir):
        #         os.makedirs(results_dir)
        #     else:
        #         print('WARNING: results_dir=%s already exists! Files might be overwritten...' % results_dir)
        self.results_dir = results_dir
        self.visualize_test_preds = visualize_test_preds
        # variables for calculating the test metrics
        self.metric_fns = [dice_score, hausdorff_score, intersection_over_union, 
                            recall_score, specificity_score, precision_score, numeric_score ]
        self.test_metrics = defaultdict(list)

        # # Get the ANIMA binaries path
        # cmd = r'''grep "^anima = " ~/.anima/config.txt | sed "s/.* = //"'''
        # self.anima_binaries_path = subprocess.check_output(cmd, shell=True).decode('utf-8').strip('\n')
        # print('ANIMA Binaries Path: ', self.anima_binaries_path)

        # Quick argument checks
        if not os.path.exists(root):
            raise ValueError('Specified path=%s for the challenge data can NOT be found!' % root)

        if len(center_crop_size) != 3:
            raise ValueError('A 3D center crop size (e.g. 512x512x512) is expected!')
        if len(subvolume_size) != 3:
            raise ValueError('A 3D subvolume size (e.g. 128x128x128) is expected!')
        if len(stride_size) != 3:
            raise ValueError('A 3D stride size (e.g. 64x64x64) is expected!')

        if any([center_crop_size[i] < subvolume_size[i] for i in range(3)]):
            raise ValueError('The center crop size must be >= subvolume size in all dimensions!')
        if any([(center_crop_size[i] - subvolume_size[i]) % stride_size[i] != 0 for i in range(3)]):
            raise ValueError('center_crop_size - subvolume_size % stride size is NOT 0 for all dimensions!')

        if not 0.0 < fraction_data <= 1.0:
            raise ValueError('`fraction_data` needs to be between 0.0 and 1.0!')
        if not 0.0 < fraction_hold_out <= 1.0:
            raise ValueError('`fraction_hold_out` needs to be between 0.0 and 1.0!')

        self.root = root
        self.center_crop_size = center_crop_size
        self.subvolume_size = subvolume_size
        self.stride_size = stride_size
        self.train = False

        # Get all subjects
        subjects_df = pd.read_csv(os.path.join(root, 'participants.tsv'), sep='\t')
        subjects = subjects_df['participant_id'].values.tolist()

        # Only use subset of the dataset if applicable (used for debugging)
        if fraction_data != 1.0:
            subjects = subjects[:int(len(subjects) * fraction_data)]

        # Hold-out a fraction of subjects for test phase
        random.seed(seed)
        random.shuffle(subjects)
        self.subjects_hold_out = subjects[:int(len(subjects) * fraction_hold_out)]
        print('Hold-out Subjects: ', self.subjects_hold_out)

        # The rest of the subjects will be used for the train and validation phases
        subjects = subjects[int(len(subjects) * fraction_hold_out):]

        # Iterate over kept subjects (i.e. after hold-out) and extract subvolumes
        self.subvolumes, self.positive_indices = [], []
        num_negatives, num_positives = 0, 0

        for subject_no, subject in enumerate(tqdm(subjects, desc='Loading Volumes')):

            # Another for loop for going through sessions
            temp_subject_path = os.path.join(root, subject)
            num_sessions_per_subject = sum(os.path.isdir(os.path.join(temp_subject_path, pth)) for pth in os.listdir(temp_subject_path))

            for ses_idx in range(1, num_sessions_per_subject+1):
                # Get paths with session numbers
                session = 'ses-0' + str(ses_idx)
                subject_images_path = os.path.join(root, subject, session, 'anat')
                subject_labels_path = os.path.join(root, 'derivatives', 'labels', subject, session, 'anat')

                # Read original subject image (i.e. 3D volume) to be used for training
                # img_path = os.path.join(subject_images_path, '%s_%s_acq-sag_T2w.nii.gz' % (subject, session))
                sag_img = nib.load(os.path.join(subject_images_path, '%s_%s_acq-sag_T2w.nii.gz' % (subject, session)))
                # hfac, wfac, dfac = sag_img.header.get_zooms()
                # Read original subject ground-truths (GT)
                gt = nib.load(os.path.join(subject_labels_path, '%s_%s_acq-sag_T2w_lesion-manual.nii.gz' % (subject, session)))

                # Check if image sizes and resolutions match
                assert sag_img.shape == gt.shape
                # print(sag_img.shape, ax_img.shape, gt.shape)  # careful! the 2nd dim is the sagittal one
                # assert np.round((sag_img.header['pixdim'].tolist()[1:4]), 2) == np.round((ax_img.header['pixdim'].tolist()[1:4]), 2) == np.round((gt.header['pixdim'].tolist()[1:4]), 2)

                # Convert to Numpy
                sag_img, gt = sag_img.get_fdata(), gt.get_fdata()

                # Apply Preprocessing steps
                # 1. center-cropping
                center_crop = CenterCrop(size=center_crop_size)
                sag_img = center_crop(sample=sag_img, metadata={'crop_params': {}})[0]
                gt = center_crop(sample=gt, metadata={'crop_params': {}})[0]

                # Get subvolumes from volumes and update the list
                sag_img_subvolumes = volume2subvolumes(volume=sag_img, subvolume_size=self.subvolume_size, stride_size=self.stride_size)
                gt_subvolumes = volume2subvolumes(volume=gt, subvolume_size=self.subvolume_size, stride_size=self.stride_size)

                assert len(sag_img_subvolumes) == len(gt_subvolumes)

                for i in range(len(sag_img_subvolumes)):
                    subvolumes_ = {
                        'sag_img': sag_img_subvolumes[i],
                        'gt': gt_subvolumes[i]}
                    self.subvolumes.append(subvolumes_)

                    # Measure positiveness based on the consensus GT
                    if np.any(gt_subvolumes[i]):
                        self.positive_indices.append(int(subject_no * len(sag_img_subvolumes) + i))
                        num_positives += 1
                    else:
                        num_negatives += 1

        self.inbalance_factor = num_negatives // num_positives
        print('Factor of overall inbalance is %d!' % self.inbalance_factor)

        print('Extracted a total of %d subvolumes!' % len(self.subvolumes))

    def __len__(self):
        return len(self.subvolumes)

    def __getitem__(self, index):
        subvolumes = self.subvolumes[index]
        sag_img_subvolume, gt_subvolume = subvolumes['sag_img'], subvolumes['gt']
        
        # Apply training augmentations
        if self.train:
            # Apply Affine Transformation with P=0.6 and Reverse with P=0.4
            if random.random() < 0.5:
                random_affine = RandomAffine(degrees=10, translate=[0.05, 0.05, 0.05], scale=[0.1, 0.1, 0.1])
                sag_img_subvolume, metadata = random_affine(sample=sag_img_subvolume, metadata={})
                gt_subvolume, _ = random_affine(sample=gt_subvolume, metadata=metadata)
                #  Elastic transform
                elastic_transform = ElasticTransform(alpha_range=[25.0, 35.0], sigma_range=[3.5, 5.5], p=1.0)
                sag_img_subvolume, metadata = elastic_transform(sample=sag_img_subvolume, metadata={})
                gt_subvolume, _ = elastic_transform(sample=gt_subvolume, metadata=metadata)
            else:
                # # Random Gamma transform
                # random_gamma = RandomGamma(log_gamma_range=[-3.0, 3.0], p=1.0)
                # sag_img_subvolume, metadata = random_gamma(sample=sag_img_subvolume, metadata={})
                # ax_img_subvolume, metadata = random_gamma(sample=ax_img_subvolume, metadata={})                
                # Random Reverse transform
                random_reverse = RandomReverse()
                sag_img_subvolume, metadata = random_reverse(sample=sag_img_subvolume, metadata={})
                gt_subvolume, _ = random_reverse(sample=gt_subvolume, metadata=metadata)
                # Random Blur
                random_blur = RandomBlur(sigma_range=[0.0, 2.0], p=1.0)
                sag_img_subvolume, metadata = random_blur(sample=sag_img_subvolume, metadata={})

        # check whether img and gt sizes are altered by any chance
        sag_img_subvolume.shape == gt_subvolume.shape == self.subvolume_size

        # Normalize to zero mean and unit variance
        normalize_instance = NormalizeInstance()
        if sag_img_subvolume.std() < 1e-5:
            sag_img_subvolume = sag_img_subvolume - sag_img_subvolume.mean()
        else:
            sag_img_subvolume, _ = normalize_instance(sample=sag_img_subvolume, metadata={})

        # Return subvolumes after converting to PyTorch tensors
        x1 = torch.tensor(sag_img_subvolume, dtype=torch.float)
        seg_y = torch.tensor(gt_subvolume, dtype=torch.float)
        # clf_y = torch.tensor(int(np.any(gt_subvolume)), dtype=torch.long)

        return index, x1, seg_y # , clf_y    

    def test(self, model):
        """Implements the test phase via animaSegPerfAnalyzer"""
        assert model is not None

        # Convert to list in case we need to assign new values in the next code block
        adjusted_center_crop_size = list(self.center_crop_size)

        # Center-crop size vs. subvolume size check to see if we need to pad the inputs or not
        for i in range(3):
            ccs_i, svs_i = adjusted_center_crop_size[i], self.subvolume_size[i]
            if ccs_i % svs_i != 0:
                # Minimally pad the initial center-crop size s.t. it becomes divisible by SV
                adjusted_center_crop_size[i] = ((ccs_i // svs_i) * svs_i) + svs_i

        # Convert back to tuple for the center-crop parameter to be immutable again
        adjusted_center_crop_size = tuple(adjusted_center_crop_size)

        # Report center-crop size parameters
        print('Original Center-Crop Size: ', self.center_crop_size)
        if adjusted_center_crop_size != self.center_crop_size:
            print('[WARNING] Adjusted Center-Crop Size: ', adjusted_center_crop_size)

        # # Compute num. subvolumes per dim and total for a quick check later on
        # num_subvolumes_per_dim = [adjusted_center_crop_size[i] // self.subvolume_size[i] for i in range(3)]
        # num_subvolumes = np.prod(num_subvolumes_per_dim)        

        for subject_no, subject in enumerate(tqdm(self.subjects_hold_out, desc='Loading Volumes')):

            # Another for loop for going through sessions
            temp_subject_path = os.path.join(self.root, subject)
            num_sessions_per_subject = sum(os.path.isdir(os.path.join(temp_subject_path, pth)) for pth in os.listdir(temp_subject_path))

            for ses_idx in range(1, num_sessions_per_subject+1):
                # Get paths with session numbers
                session = 'ses-0' + str(ses_idx)
                subject_images_path = os.path.join(self.root, subject, session, 'anat')
                subject_labels_path = os.path.join(self.root, 'derivatives', 'labels', subject, session, 'anat')

                # Read original subject image (i.e. 3D volume) to be used for training
                sag_img = nib.load(os.path.join(subject_images_path, '%s_%s_acq-sag_T2w.nii.gz' % (subject, session)))
                # hfac, wfac, dfac = sag_img.header.get_zooms()
                # Read original subject ground-truths (GT)
                gt = nib.load(os.path.join(subject_labels_path, '%s_%s_acq-sag_T2w_lesion-manual.nii.gz' % (subject, session)))

                # Check if image sizes and resolutions match
                assert sag_img.shape == gt.shape
                # assert sag_img.header['pixdim'].tolist() == ax_img.header['pixdim'].tolist() == gt.header['pixdim'].tolist()

                # Convert to Numpy
                sag_img, gt = sag_img.get_fdata(), gt.get_fdata()

                # Apply center-cropping
                center_crop = CenterCrop(size=self.center_crop_size)
                sag_img = center_crop(sample=sag_img, metadata={'crop_params': {}})[0]
                gt = center_crop(sample=gt, metadata={'crop_params': {}})[0]

                # Get subvolumes from volumes
                # NOTE: We use subvolume size for the stride size to get non-overlapping test subvolumes
                sag_img_subvolumes = volume2subvolumes(volume=sag_img, subvolume_size=self.subvolume_size, stride_size=self.subvolume_size)
                gt_subvolumes = volume2subvolumes(volume=gt, subvolume_size=self.subvolume_size, stride_size=self.subvolume_size)
                assert len(sag_img_subvolumes) == len(gt_subvolumes)

                # Collect individual subvolume predictions for full volume segmentation (i.e. full scan for one subject)
                pred_subvolumes = []
                for i in range(len(sag_img_subvolumes)):
                    sag_img_subvolume, gt_subvolume = sag_img_subvolumes[i], gt_subvolumes[i]
                    # Normalize images to zero mean and unit variance
                    normalize_instance = NormalizeInstance()
                    if sag_img_subvolume.std() < 1e-5:
                        # If subvolumes uniform: do mean-subtraction
                        sag_img_subvolume = sag_img_subvolume - sag_img_subvolume.mean()
                    else:
                        sag_img_subvolume, _ = normalize_instance(sample=sag_img_subvolume, metadata={})
                    # twice unsqueezed such that batch_size=1 and num_channels=1
                    x1 = torch.tensor(sag_img_subvolume, dtype=torch.float).view(1, 1, *sag_img_subvolume.shape)

                    # Get the standard subvolume prediction
                    seg_y_hat = model(x1).squeeze().detach().cpu().numpy()

                    pred_subvolumes.append(seg_y_hat)

                # Convert the list of subvolume predictions to a single volume segmentation / prediction
                pred = subvolumes2volume(subvolumes=pred_subvolumes, volume_size=adjusted_center_crop_size)
                assert pred.shape == gt.shape

                # Apply the original center-crop size in case it was adjusted before
                if pred.shape != self.center_crop_size:
                    gt_sum_before_crop = np.sum(gt)
                    center_crop = CenterCrop(size=self.center_crop_size)
                    pred = center_crop(sample=pred, metadata={'crop_params': {}})[0]
                    gt = center_crop(sample=gt, metadata={'crop_params': {}})[0]

                    # Check if padding & un-padding removes any lesion GTs; only continue if it does not
                    if abs(np.sum(gt) - gt_sum_before_crop) > 1e-6:
                        # NOTE: Apparently np.sum() can have epsilon differences even with same values!
                        raise ValueError('Padding & un-padding cropped out lesions! Check your center-crop parameters!')
            
                # Calculate the test metrics on the "soft" prediction only
                for metric_fn in self.metric_fns:
                    res = metric_fn(pred, gt)
                    dict_key = metric_fn.__name__
                    self.test_metrics[dict_key].append(res)

                # Save the prediction and the center-cropped GT as new NIfTI files
                pred_nib = nib.Nifti1Image(pred, affine=np.eye(4))
                gt_nib = nib.Nifti1Image(gt, affine=np.eye(4))
                nib.save(img=pred_nib, filename=os.path.join(self.results_dir, '%s_%s_pred.nii.gz' % (subject, session)))
                nib.save(img=gt_nib, filename=os.path.join(self.results_dir, '%s_%s_gt.nii.gz' % (subject, session)))            
        
        print()
        # Print the mean and std for each metric
        for key in self.test_metrics:
            print("\t%s --> Mean: %0.3f, Std: %0.3f" % (key, np.mean(self.test_metrics[key]), np.std(self.test_metrics[key])))
            with open(os.path.join(self.results_dir, 'log.txt'), 'a') as f:
                print("\t%s --> Mean: %0.3f, Std: %0.3f" % (key, np.mean(self.test_metrics[key]), np.std(self.test_metrics[key])), file=f)

  


if __name__ == "__main__":
    dataroot = "/home/nagakarthik/deepLearning/datasets/sci-zurich_preprocessed_no-json"
    fd = 1.0
    fho = 0.2
    dataset = SCIZurichDataset(dataroot, fraction_data=fd, fraction_hold_out=fho)
    # subjects_train_val = int((fd-fho)*len(dataset))
    train_size = int(0.6 * len(dataset))
    valid_size = len(dataset) - train_size 
    train_data, valid_data = random_split(dataset, lengths=[train_size, valid_size])
    print(len(train_data), len(valid_data))

