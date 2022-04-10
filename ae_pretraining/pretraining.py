# utility packages
from multiprocessing import reduction
import os
from datetime import datetime
import argparse

import numpy as np
import matplotlib.pyplot as plt
timestamp = f'{datetime.now():%Y%m%d-%H%M}'

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
from datasets_sc_ae import SCIZurichDataset
from unet3d_ae import ModifiedUNet3DEncoder, ModifiedUNet3DDecoder
from sim_unet3d_ae import UNet3DEncoder, UNet3DDecoder
from utils import save_wandb_img

grp_name = 'ae-pretraining'

parser = argparse.ArgumentParser(description='Script for pre-training custom models for SCI Lesion Segmentation.')
# Arguments for model, data, and training and saving
parser.add_argument('-m', '--model_type', choices=['unet'], default='unet', type=str, help='Model type to be used')
# dataset
parser.add_argument('-dr', '--dataset_root', 
                    default='/home/GRAMES.POLYMTL.CA/u114716/duke/temp/muena/sci-zurich_preprocessed', type=str,
                    help='Root path to the BIDS- and ivadomed-compatible dataset')
parser.add_argument('-fd', '--fraction_data', default=1.0, type=float, help='Fraction of data to use. Helps with debugging.')
parser.add_argument('-fho', '--fraction_hold_out', default=0.1, type=float, help='Fraction of data to hold-out of for the test phase')
parser.add_argument('-ftv', '--fraction_train_val', nargs='+', default=[0.75, 0.15], 
                    help="Train and validation split. Should sum to (fraction_data - fraction_hold_out)")
# model 
parser.add_argument('-t', '--task', choices=['sc', 'mc'], default='sc', type=str, help="Single-channel or Multi-channel model ")
parser.add_argument('-bnf', '--base_n_filter', default=8, type=int, help="Number of Base Filters")
parser.add_argument('-dep', '--depth', default=4, type=int, help="Depth of Simple UNet3D")
parser.add_argument('-ccs', '--center_crop_size', nargs='+', default=[128, 256, 96], help='List containing center crop size for preprocessing')
parser.add_argument('-svs', '--subvolume_size', nargs='+', default=[128, 256, 96], help='List containing subvolume size')
parser.add_argument('-srs', '--stride_size', nargs='+', default=[128, 256, 96], help='List containing stride size')
# optimizations
parser.add_argument('-p', '--precision', default=32, type=int, help='Precision for training')
parser.add_argument('-ne', '--num_epochs', default=500, type=int, help='Number of epochs for the training process')
parser.add_argument('-bs', '--batch_size', default=16, type=int, help='Batch size of the training and validation processes')
parser.add_argument('-nw', '--num_workers', default=4, type=int, help='Number of workers for the dataloaders')
parser.add_argument('-lr', '--learning_rate', default=5e-4, type=float, help='Learning rate for training the model')
parser.add_argument('-wd', '--weight_decay', type=float, default=0.01, help='Weight decay (i.e. regularization) value in AdamW')
parser.add_argument('-pat', '--patience', default=20, type=int, help='number of validation steps (val_every_n_iters) to wait before early stopping')
# parser.add_argument('--T_0', default=100, type=int, help='number of steps in each cosine cycle')
parser.add_argument('-epb', '--enable_progress_bar', default=False, type=bool, help='by default is disabled since it doesnt work in colab')
# parser.add_argument('--val_every_n_iters', default='100', type=int, help='num of iterations before validation')
parser.add_argument('-gpus', '--num_gpus', default=1, type=int, help="Number of GPUs to use")

# saving
parser.add_argument('-mn', '--model_name', default='unet3d', type=str, help='Model ID to-be-used for saving the .pt saved model file')
parser.add_argument('-s', '--save', default='saved_models', type=str, help='Path to the saved models directory')
parser.add_argument('-c', '--continue_from_checkpoint', default=False, action='store_true', help='Load model from checkpoint and continue training')
parser.add_argument('-se', '--seed', default=42, type=int, help='Set seeds for reproducibility')
parser.add_argument('-v', '--visualize_test_preds', default=False, action='store_true',
                    help='Enable to save subvolume predictions during the test phase for visual assessment')

cfg = parser.parse_args()
model_id = '%s-t=%s-ccs=%d-svs=%s-bs=%d-lr=%s-bnf=%d-se=%d' % \
            (cfg.model_name, cfg.task, cfg.center_crop_size[1], str(cfg.subvolume_size[1]),
            cfg.batch_size, str(cfg.learning_rate), cfg.base_n_filter, cfg.seed)
print('MODEL ID: %s' % model_id)

# define the dataset
dataset = SCIZurichDataset(
    root=cfg.dataset_root,
    fraction_data=cfg.fraction_data,
    fraction_hold_out=cfg.fraction_hold_out,
    center_crop_size=(int(cfg.center_crop_size[0]), int(cfg.center_crop_size[1]), int(cfg.center_crop_size[2])),
    subvolume_size=(int(cfg.subvolume_size[0]), int(cfg.subvolume_size[1]), int(cfg.subvolume_size[2])),
    stride_size=(int(cfg.stride_size[0]), int(cfg.stride_size[1]), int(cfg.stride_size[2])),
    results_dir='outputs/%s_RESULTS' % (model_id),
    visualize_test_preds=cfg.visualize_test_preds,
    seed=cfg.seed)

class SCISegModel(pl.LightningModule):
    def __init__(self, cfg):
        super(SCISegModel, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.init_timestamp = f'{datetime.now(): %Y%m%d-%H%M%S}'
        self.every_n_epochs = 50

        # # Configure saved models directory
        # if not os.path.isdir(self.cfg.save):
        #     os.makedirs(self.cfg.save)

        # instantiate model and datasets
        if self.cfg.model_name == 'unet3d':
            self.encoder_net = ModifiedUNet3DEncoder(in_channels=1, base_n_filter=self.cfg.base_n_filter)
            self.decoder_net = ModifiedUNet3DDecoder(n_classes=1, base_n_filter=self.cfg.base_n_filter)
        elif self.cfg.model_name == 'simple-unet3d':
            self.encoder_net = UNet3DEncoder(n_channels=1 if self.cfg.task == 'sc' else 2, init_filters=self.cfg.base_n_filter)
            self.decoder_net = UNet3DDecoder(init_filters=self.cfg.base_n_filter, n_classes=1)
            
        # Data split for train and validation phases
        train_size = int(self.cfg.fraction_train_val[0] * len(dataset)) 
        valid_size = len(dataset) - train_size
        self.train_dataset, self.valid_dataset = random_split(dataset, lengths=[train_size, valid_size])

        # Define loss function
        self.rec_criterion = nn.MSELoss(reduction='none')

        # # for logging/visualizing
        # self.loss_visualization_step = 0.5
        # self.best_valid_loss, self.best_train_loss = 1.0, 1.0
        self.train_losses, self.valid_losses = [], []
        # # self.num_iters_per_epoch = int(np.ceil(len(self.train_dataset) / self.cfg.batch_size))

    def forward(self, x):
        pass

    def compute_reconstruction_loss(self, batch):
        """ Loss function for training the model """
        if self.cfg.task == 'mc':       # multi-channel model
            dataset_indices, x1, x2, seg_y = batch
            x1, x2 = x1.unsqueeze(1), x2.unsqueeze(1)    # add channel dimension for concatenation
            seg_y_hat = self.net(x1, x2)
        elif self.cfg.task == 'sc':     # single-channel model
            _, x, _ = batch
            x = x.unsqueeze(1)
            x_latent, context_feats = self.encoder_net(x)
            x_hat = self.decoder_net(x_latent, context_feats)            
        
        reconstruction_loss = self.rec_criterion(x_hat, x)
        reconstruction_loss = reconstruction_loss.sum(dim=[1,2,3,4]).mean(dim=0)
        
        return x_hat, reconstruction_loss

    def training_step(self, batch, batch_idx):
        preds, loss = self.compute_reconstruction_loss(batch)
        # self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.train_losses.append(loss.item())

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        train_loss = np.mean(self.train_losses)
        # self.log('train_loss', avg_train_loss, on_step=False, on_epoch=True)
        self.log('train_loss', train_loss, on_step=False, on_epoch=True)
        self.train_losses = []

    def validation_step(self, batch, batch_idx):
        if self.cfg.task == 'mc':       # multi-channel model
            _, img1, img2, gts = batch
        elif self.cfg.task == 'sc':
            _, img1, _ = batch
        preds, loss = self.compute_reconstruction_loss(batch)
        self.valid_losses.append(loss.item())
        # self.log('valid_loss', loss, on_step=False, on_epoch=True)
        
        # qualitative results on wandb 
        if (self.current_epoch+1) % self.every_n_epochs == 0:
            save_wandb_img("Validation", img1.unsqueeze(1), preds)
    
    def validation_epoch_end(self, outputs):
        valid_loss = np.mean(self.valid_losses)
        self.log('valid_loss', valid_loss, on_step=False, on_epoch=True)
        self.valid_losses = []
                        
    def configure_optimizers(self):
        optimizer = optim.AdamW(params=self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        # optimizer = optim.Adam(params=self.parameters(), lr=self.cfg.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=self.cfg.patience, min_lr=5e-5)
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.cfg.T_0, eta_min=1e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_loss"}
        # return [optimizer], [scheduler]
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.cfg.batch_size, pin_memory=True,
                             shuffle = True, drop_last=True, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size = self.cfg.batch_size, pin_memory=True,
                             shuffle = False, drop_last=False, num_workers=self.cfg.num_workers)


def main(cfg):
    # experiment tracker (you need to sign in with your account)
    wandb_logger = pl.loggers.WandbLogger(
                            name='%s-t=%s<-%s' % (cfg.model_name, cfg.task, timestamp), 
                            group= '%s'%(grp_name), 
                            log_model=True, # save best model using checkpoint callback
                            project='sci-zurich',
                            config=cfg)

    # to save the best model on validation
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=cfg.save, filename=model_id, monitor="valid_loss", 
                                                save_top_k=1, mode="min", save_last=False)
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

    trainer.fit(model)
    print("------- Training Done! -------")

    print("------- Loading the Best Model! ------")     # the PyTorch Lightning way
    # load the best checkpoint after training
    loaded_model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, strict=False)
    pretrained_encoder = loaded_model.encoder_net
    
    folder_name = "./" + "saved_enc_models/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    save_path = folder_name + "best_enc_" + cfg.model_id + ".pt"   
    torch.save(pretrained_encoder.state_dict(), save_path)
        
        # # Evaluate on the test subjects right after training
        # print("------- Testing Begins! -------")
        # dataset.train = False
        # dataset.test(loaded_model.net)
        # print()
        # print("------- Testing Done! -------")

if __name__ == '__main__':
    main(cfg)