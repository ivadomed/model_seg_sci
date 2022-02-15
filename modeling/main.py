# utility packages
import os
from datetime import datetime
import argparse

import numpy as np
import matplotlib.pyplot as plt
timestamp = f'{datetime.now():%Y%m%d-%H%M%S}'

# machine learning packages
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, dataloader, random_split
import torchvision.transforms as transforms

# dataloaders and segmentation models
from datasets import SCIZurichDataset
from unet3d import ModifiedUNet3D
from ivadomed.losses import DiceLoss

grp_name = 'SagT2w-training'

parser = argparse.ArgumentParser(description='Script for training custom models for SCI Lesion Segmentation.')
# Arguments for model, data, and training and saving
parser.add_argument('-e', '--only_eval', default=False, action='store_true', help='Only do evaluation, i.e. skip training!')
parser.add_argument('-m', '--model_type', choices=['unet'], default='unet', type=str, help='Model type to be used')
# dataset
parser.add_argument('-dr', '--dataset_root', 
                    default='/home/GRAMES.POLYMTL.CA/u114716/duke/temp/muena/sci-zurich_preprocessed_no-json', type=str,
                    help='Root path to the BIDS- and ivadomed-compatible dataset')
parser.add_argument('-fd', '--fraction_data', default=1.0, type=float, help='Fraction of data to use. Helps with debugging.')
parser.add_argument('-fho', '--fraction_hold_out', default=0.2, type=float, help='Fraction of data to hold-out of for the test phase')
parser.add_argument('-ftv', '--fraction_train_val', nargs='+', default=[0.6, 0.2], 
                    help="Train and validation split. Should sum to (fraction_data - fraction_hold_out)")
# model 
parser.add_argument('-t', '--task', choices=['sc', 'mc'], default='sc', type=str, help="Single- or Multi-contrast model ")
parser.add_argument('-bnf', '--base_n_filter', default=8, type=int, help="Number of Base Filters")
parser.add_argument('-svs', '--subvolume_size', nargs='+', default=[96, 48, 32], help='List containing subvolume size per dimension')
parser.add_argument('-srs', '--stride_size', nargs='+', default=[96, 48, 32], help='List containing stride size per dimension')
# optimization
parser.add_argument('-p', '--precision', default=32, type=int, help='Precision for training')
parser.add_argument('-ne', '--num_epochs', default=500, type=int, help='Number of epochs for the training process')
parser.add_argument('-bs', '--batch_size', default=16, type=int, help='Batch size of the training and validation processes')
parser.add_argument('-nw', '--num_workers', default=4, type=int, help='Number of workers for the dataloaders')
parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='Learning rate for training the model')
parser.add_argument('-wd', '--weight_decay', type=float, default=0.01, help='Weight decay (i.e. regularization) value in AdamW')
parser.add_argument('-pat', '--patience', default=50, type=int, help='number of validation steps (val_every_n_iters) to wait before early stoping')
parser.add_argument('--T_0', default=500, type=int, help='number of steps in each cosine cycle')
parser.add_argument('-epb', '--enable_progress_bar', default=False, type=bool, help='by default is disabled since it doesnt work in colab')
# parser.add_argument('--val_every_n_iters', default='100', type=int, help='num of iterations before validation')
parser.add_argument('-gpus', '--num_gpus', default=1, type=int, help="Number of GPUs to use")
# parser.add_argument('-bal', '--balance_strategy', choices=['none', 'naive_duplication', 'cheap_duplication', 'naive_removal'], default='naive_duplication', type=str,
#                     help='The balancing strategy to employ for the training subset')

# parser.add_argument('-mcd', '--mc_dropout', default=False, action='store_true', help='To use Monte Carlo samples for validation and testing')
# parser.add_argument('-n_mc', '--num_mc_samples', default=0, type=int, help="Number of MC samples to use")
# saving
parser.add_argument('-id', '--model_id', default='unet3d', type=str, help='Model ID to-be-used for saving the .pt saved model file')
parser.add_argument('-s', '--save', default='./saved_models', type=str, help='Path to the saved models directory')
parser.add_argument('-c', '--continue_from_checkpoint', default=False, action='store_true', help='Load model from checkpoint and continue training')
parser.add_argument('-se', '--seed', default=42, type=int, help='Set seeds for reproducibility')
parser.add_argument('-v', '--visualize_test_preds', default=False, action='store_true',
                    help='Enable to save subvolume predictions during the test phase for visual assessment')

cfg = parser.parse_args()

# define the dataset
dataset = SCIZurichDataset(
    root=cfg.dataset_root,
    fraction_data=cfg.fraction_data,
    fraction_hold_out=cfg.fraction_hold_out,
    center_crop_size=(384, 96, 64),
    subvolume_size=(cfg.subvolume_size[0], cfg.subvolume_size[1], cfg.subvolume_size[2]),
    stride_size=(cfg.stride_size[0], cfg.stride_size[1], cfg.stride_size[2]),
    results_dir='%s-%s_RESULTS' % (cfg.model_id, timestamp),
    visualize_test_preds=cfg.visualize_test_preds,
    seed=cfg.seed)

class SCISegModel(pl.LightningModule):
    def __init__(self, cfg):
        super(SCISegModel, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.init_timestamp = f'{datetime.now(): %Y%m%d-%H%M%S}'

        model_id = '%s-t=%s <-%s' % (self.cfg.model_id, self.cfg.task, self.init_timestamp)
        print('MODEL ID: %s' % model_id)

        # Configure saved models directory
        if not os.path.isdir(self.cfg.save):
            os.makedirs(self.cfg.save)

        # # TODO: Revisit saving path and save file names
        # if not self.cfg.only_eval:
        #     print('Trained model will be saved to: %s' % os.path.join(self.cfg.save, '%s.pt' % model_id))
        # else:
        #     print('Running Evaluation: (1) Loss metrics on validation set, and (2) ANIMA metrics on test set')
        #     self.cfg.continue_from_checkpoint = True

        # instantiate model and datasets
        self.net = ModifiedUNet3D(cfg=self.cfg)
        # not defining inside the class because need to use it for testing
        # self.dataset = SCIZurichDataset(
        #     root=self.cfg.dataset_root,
        #     fraction_data=self.cfg.fraction_data,
        #     fraction_hold_out=self.cfg.fraction_hold_out,
        #     center_crop_size=(384, 96, 32),
        #     subvolume_size=(self.cfg.subvolume_size[0], self.cfg.subvolume_size[1], self.cfg.subvolume_size[2]),
        #     stride_size=(self.cfg.stride_size[0], self.cfg.stride_size[1], self.cfg.stride_size[2]),
        #     results_dir='%s_RESULTS' % model_id,
        #     visualize_test_preds=self.cfg.visualize_test_preds,
        #     seed=self.cfg.seed)
        
        # Data split for train and validation phases
        train_size = int(self.cfg.fraction_train_val[0] * len(dataset)) 
        valid_size = len(dataset) - train_size
        self.train_dataset, self.valid_dataset = random_split(dataset, lengths=[train_size, valid_size])

        # Define loss function
        self.seg_criterion = DiceLoss(smooth=1.0)

        # for logging/visualizing
        self.loss_visualization_step = 0.1
        self.best_valid_loss, self.best_train_loss = 1.0, 1.0
        self.train_losses, self.valid_losses = [], []
        self.num_iters_per_epoch = int(np.ceil(len(self.train_dataset) / self.cfg.batch_size))

    def forward(self, x):
        pass

    def compute_loss(self, batch):
        """ Loss function for training the model """
        dataset_indices, x1, seg_y = batch
        x1 = x1.unsqueeze(1)    # add channel dimension
        seg_y_hat = self.net(x1)
        seg_loss = self.seg_criterion(seg_y_hat, seg_y)
        return seg_loss

    def training_step(self, batch, batch_idx):
        dataset.train = True
        _, imgs, _ = batch
        loss = self.compute_loss(batch)
        self.train_losses += [loss.item()] * len(imgs)

        if loss < self.best_train_loss - self.loss_visualization_step and batch_idx==0:
          self.best_train_loss = loss.item()

        return loss

    def validation_step(self, batch, batch_idx):
        dataset.train = False
        _, imgs, _ = batch
        loss = self.compute_loss(batch)
        self.valid_losses += [loss.item()] * len(imgs)

        # qualitative results on wandb when first batch dice improves by 10%
        if loss < self.best_valid_loss - self.loss_visualization_step and batch_idx==0:
          self.best_valid_loss = loss.item()

    def on_train_epoch_end(self):
        train_loss = np.mean(self.train_losses)
        self.log('train_loss', train_loss)
        self.train_losses = []

    def on_validation_epoch_end(self):
        valid_loss = np.mean(self.valid_losses)
        self.log('valid_loss', valid_loss)
        self.valid_losses = []

    def configure_optimizers(self):
        optimizer = optim.AdamW(params=self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, eta_min=1e-5)
        
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.cfg.batch_size, pin_memory=True,
                             shuffle = True, drop_last=True, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size = self.cfg.batch_size, pin_memory=True,
                             shuffle = False, drop_last=False, num_workers=self.cfg.num_workers)


def main(cfg):
    # experiment tracker (you need to sign in with your account)
    wandb_logger = pl.loggers.WandbLogger(
                            name='%s-t=%s<-%s' % (cfg.model_id, cfg.task, timestamp), 
                            group= '%s'%(grp_name), 
                            log_model=True, # save best model using checkpoint callback
                            project='sci-zurich',
                            config=cfg)

    # to save the best model on validation
    checkpoint = pl.callbacks.ModelCheckpoint(
        filename="best_model_"+str(timestamp), 
        monitor=None, save_top_k=1, mode="min", save_last=False)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    early_stop = pl.callbacks.EarlyStopping(monitor="valid_loss", min_delta=0.00, 
                                            patience=cfg.patience, verbose=False, mode="min")

    model = SCISegModel(cfg)
    trainer = pl.Trainer(
        default_root_dir=os.getcwd(),
        devices=cfg.num_gpus, accelerator="gpu", strategy="ddp",
        logger=wandb_logger, 
        callbacks=[checkpoint, lr_monitor, early_stop],
        max_epochs=cfg.num_epochs, 
        precision=cfg.precision,
        enable_progress_bar=cfg.enable_progress_bar)
        # check_val_every_n_epoch=(cfg.val_every_n_iters//model.num_iters_per_epoch))

    # log gradients, parameter histogram and model topology
    wandb_logger.watch(model)
    
    trainer.fit(model)
    print("------- Training Done! -------")

    print("------- Loading the Best Model! ------")     # the PyTorch Lightning way
    # load the best checkpoint after training
    print(trainer.checkpoint_callback.best_model_path)
    loaded_model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, strict=False)
    # print(loaded_model)   # prints the whole SCISegModel class

    print("------- Testing Begins! -------")
    dataset.train = False
    dataset.test(loaded_model.net)
    print()
    print("------- Testing Done! -------")

if __name__ == '__main__':
    main(cfg)