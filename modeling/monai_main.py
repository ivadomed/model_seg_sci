import os
from secrets import choice
import shutil
import tempfile
import argparse

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import wandb
import torch
import pytorch_lightning as pl

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss, DiceLoss, MaskedDiceLoss, FocalLoss
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet, DynUNet, BasicUNet, SegResNet, UNETR
from monai.data import (DataLoader, CacheDataset, load_decathlon_datalist, decollate_batch, list_data_collate)
from monai.transforms import (AsDiscrete, AddChanneld, Compose, CropForegroundd, LoadImaged, Orientationd, RandFlipd, 
                    RandCropByPosNegLabeld, RandShiftIntensityd, ScaleIntensityRanged, Spacingd, RandRotate90d, ToTensord,
                    SpatialPadd, NormalizeIntensityd, EnsureType, RandScaleIntensityd, RandWeightedCropd, EnsureChannelFirstd,
                    AsDiscreted, RandSpatialCropSamplesd, HistogramNormalized)


# print_config()

def plot_images_and_labels():
    
    case_num = 0
    print(len(train_ds[case_num]))

    if len(train_ds[case_num]) == 4:
        
        for i in range(len(train_ds[case_num])):
            
            img_name = os.path.split(train_ds[case_num][i]["image_meta_dict"]["filename_or_obj"])[1]
            # print(os.path.split(val_ds[case_num]["image_meta_dict"]["filename_or_obj"]))
            img = train_ds[case_num][i]["image"]
            label = train_ds[case_num][i]["label"]
            slice_num = img.shape[1]//2

            slice_map = {
                "sub-zh03_ses-01_acq-sag_T2w.nii.gz": slice_num,
                "sub-zh70_ses-01_acq-sag_T2w.nii.gz": slice_num,
                "sub-zh80_ses-01_acq-sag_T2w.nii.gz": slice_num,
                "sub-zh44_ses-01_acq-sag_T2w.nii.gz": slice_num,
                "sub-zh53_ses-01_acq-sag_T2w.nii.gz": slice_num,
            }

            print(f"image shape: {img.shape}, label shape: {label.shape}")
            plt.figure("image", (12, 6))
            plt.subplot(1, 2, 1)
            plt.title(f"image_{i}")
            plt.imshow(img[0, slice_map[img_name], :, :].detach().cpu(), cmap="gray")
            plt.subplot(1, 2, 2)
            plt.title(f"label_{i}")
            plt.imshow(label[0, slice_map[img_name], :, :].detach().cpu())
            plt.tight_layout()        
            plt.show()
    
    else:
        img_name = os.path.split(train_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
        # print(os.path.split(val_ds[case_num]["image_meta_dict"]["filename_or_obj"]))
        img = train_ds[case_num]["image"]
        label = train_ds[case_num]["label"]
        slice_num = img.shape[1]//2

        slice_map = {
            "sub-zh03_ses-01_acq-sag_T2w.nii.gz": slice_num,
            "sub-zh70_ses-01_acq-sag_T2w.nii.gz": slice_num,
            "sub-zh80_ses-01_acq-sag_T2w.nii.gz": slice_num,
            "sub-zh44_ses-01_acq-sag_T2w.nii.gz": slice_num,
            "sub-zh53_ses-01_acq-sag_T2w.nii.gz": slice_num,
        }

        print(f"image shape: {img.shape}, label shape: {label.shape}")
        plt.figure("image", (18, 6))
        plt.subplot(1, 2, 1)
        plt.title("image")
        plt.imshow(img[0, slice_map[img_name], :, :].detach().cpu(), cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title("label")
        plt.imshow(label[0, slice_map[img_name], :, :].detach().cpu())
        plt.show()



# create a "model"-agnostic class with PL to use different models on both datasets
class Model(pl.LightningModule):
    def __init__(self, args, data_root, net, loss_function, optimizer_class):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        
        self.root = data_root
        self.lr = args.learning_rate
        self.net = net
        self.loss_function = loss_function
        self.optimizer_class = optimizer_class

        self.best_val_dice, self.best_val_epoch = 0, 0
        # self.check_val = args.check_val_every_n_epochs
        self.warmup_epochs = 20
        self.metric_values = []
        self.epoch_loss_values = []

        # define cropping and padding dimensions
        self.voxel_cropping_size = (64, 64, 128)
        self.spatial_padding_size = (64, 64, 300)
        self.inference_roi_size = (64, 64, 128)

        # define post-processing transforms
        self.post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
        
        # define evaluation metric
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    def forward(self, x):
        return self.net(x)

    def prepare_data(self):
        split_JSON = "dataset_0.json"
        datasets = self.root + split_JSON

        # load datasets (they are in Decathlon datalist format)
        datalist = load_decathlon_datalist(datasets, True, "training")
        val_files = load_decathlon_datalist(datasets, True, "validation")
        test_files = load_decathlon_datalist(datasets, True, "test")

        # define training and validation transforms
        train_transforms = Compose([   
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            # AsDiscreted(keys=["label"], threshold=0.3),     # zurich labels are soft, need to convert to binary for PosNegCropping
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"],pixdim=(0.75, 0.75, 0.75), mode=("bilinear", "nearest"),),
            SpatialPadd(keys=["image", "label"], spatial_size=self.spatial_padding_size, mode="minimum"),
            # CropForegroundd(keys=["image", "label"], source_key="image"),     # crops >0 values with a bounding box
            # RandCropByPosNegLabeld(
            #     keys=["image", "label"], label_key="label", 
            #     spatial_size=(64, 64, 128),
            #     pos=3, neg=1, 
            #     num_samples=4, 
            #     image_key="image", image_threshold=0.),   # error raised because of the soft GT, no resulting the same size of flattened matrices
            RandWeightedCropd(keys=["image", "label"], w_key="label", spatial_size=self.voxel_cropping_size, num_samples=4),
            # RandSpatialCropSamplesd(keys=["image", "label"], roi_size=voxel_size, num_samples=4),
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.50,),
            RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.50,),
            RandFlipd(keys=["image", "label"],spatial_axis=[2],prob=0.50,),
            RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3,),
            NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
            # RandShiftIntensityd(keys=["image"], offsets=0.10, prob=1.0,),
            # RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            HistogramNormalized(keys="image", mask=None),
            ToTensord(keys=["image", "label"]), 
        ])

        val_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"],pixdim=(0.75, 0.75, 0.75), mode=("bilinear", "nearest"),),
            SpatialPadd(keys=["image", "label"], spatial_size=self.spatial_padding_size, mode="minimum"),
            # ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            # CropForegroundd(keys=["image", "label"], source_key="image"),
            NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
            HistogramNormalized(keys="image", mask=None),
            ToTensord(keys=["image", "label"]),
        ])

        # TODO: define test_transforms

        # define training and validation dataloaders
        self.train_ds = CacheDataset(data=datalist, transform=train_transforms, cache_rate=1.0, num_workers=8)
        self.val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=12, cache_rate=1.0, num_workers=4)
        # print(len(train_loader))
        # TODO: define test cache dataset


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=4, pin_memory=True, 
                            collate_fn=list_data_collate)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr, weight_decay=1e-5)
        # TODO: add lr_scheduler
        return optimizer

    def _compute_loss(self, preds, labels):

        if self.args.dataset == 'zurich':
            # convert labels to binary masks to calculate the weights for CELoss
            # and to convert to one-hot encoding for DiceLoss
            labels = (labels > 0.3).long()
        # print(f"labels binary: {labels}")

        # # define loss functions
        # if self.loss_function == 'dice':
        #     criterion = DiceLoss(to_onehot_y=True, softmax=True)
        # elif self.loss_function in ['dice_ce', 'dice_ce_sq']:                
        #     # compute weights for cross entropy loss
        #     labels_b = labels[0]
        #     normalized_ce_weights = get_ce_weights(labels_b.detach().cpu())
        #     # print(f"normed ce weights: {normalized_ce_weights}")
        #     if self.loss_function == 'dice_ce':
        #         criterion = DiceCELoss(to_onehot_y=True, softmax=True, ce_weight=normalized_ce_weights)
        #     else:
        #         criterion = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, ce_weight=normalized_ce_weights)
        # loss = criterion(preds, labels)
        
        loss = self.loss_function(preds, labels)

        return loss

    def training_step(self, batch, batch_idx):
        inputs, labels = (batch["image"], batch["label"])
        output = self.forward(inputs)

        # calculate training loss
        loss = self._compute_loss(output, labels)
        
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log('train_loss', avg_loss, on_step=False, on_epoch=True)

        self.epoch_loss_values.append(avg_loss.detach().cpu().numpy())

    
    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        inference_roi_size = self.inference_roi_size
        sw_batch_size = 4
        outputs = sliding_window_inference(images, inference_roi_size, sw_batch_size, 
                                            self.forward, padding_mode="reflect")
        
        # calculate validation loss
        loss = self._compute_loss(outputs, labels)
        
        # post-process for calculating the evaluation metric
        post_outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        post_labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=post_outputs, y=post_labels)
        
        return {
            "val_loss": loss, 
            "val_number": len(post_outputs),
            "preds": outputs,
            "images": images,
            "labels": labels
        }

    def validation_epoch_end(self, outputs):
        val_loss, num_val_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_val_items += output["val_number"]
        
        mean_val_loss = torch.tensor(val_loss / num_val_items)

        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        wandb_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

        # qualitative results on wandb
        for output in outputs:
            fig = visualize(preds=output["preds"], imgs=output["images"], gts=output["labels"], num_slices=7)
            wandb.log({"Validation Output Visualizations": fig})
            plt.close()

        print(
            f"Current epoch: {self.current_epoch}"
            f"\nCurrent Mean Dice: {mean_val_dice:.4f}"
            f"\nBest Mean Dice: {self.best_val_dice:.4f} at Epoch: {self.best_val_epoch}"
            f"\n----------------------------------------------------")
        
        self.metric_values.append(mean_val_dice)

        # log on to wandb
        self.log_dict(wandb_logs)
        
        return {"log": wandb_logs}

        

# dummy_transforms = Compose([   
#         LoadImaged(keys=["image", "label"]),
#         AddChanneld(keys=["image", "label"]),
#         Orientationd(keys=["image", "label"], axcodes="RAS"),
#         Spacingd(keys=["image", "label"],pixdim=(0.75, 0.75, 0.75), mode=("bilinear", "nearest"),),
#         SpatialPadd(keys=["image", "label"], spatial_size=spatial_padding_size),
#         # CropForegroundd(keys=["image", "label"], source_key="image"),     # crops >0 values with a bounding box
#         RandCropByPosNegLabeld(
#             keys=["image", "label"], label_key="label", 
#             spatial_size=(64, 64, 128),
#             pos=3, neg=1, 
#             num_samples=4, 
#             image_key="image", image_threshold=0.1),
#         # RandWeightedCropd(keys=["image", "label"], w_key="label", spatial_size=(64, 64, 128), num_samples=4),
#         # RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.50,),
#         # RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.50,),
#         # RandFlipd(keys=["image", "label"],spatial_axis=[2],prob=0.50,),
#         # RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3,),
#         NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
#         RandShiftIntensityd(keys=["image"], offsets=0.10, prob=1.0,),
#         # RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
#         ToTensord(keys=["image", "label"]),
#     ])


# for case_num in range(len(train_loader)//4):
#     # img = train_ds[case_num][0]["image"]
#     # label = train_ds[case_num][0]["label"]

#     # print(len(train_ds[case_num]))

#     if len(train_ds[case_num]) == 4:
#         img = train_ds[case_num][0]["image"]
#         label = train_ds[case_num][0]["label"]
#     else:
#         img = train_ds[case_num]["image"]
#         label = train_ds[case_num]["label"]
#     # print(f"image shape: {img.shape}, label shape: {label.shape}")
#     # print(f"img min: {img.min()} \t img max: {img.max()}")
#     # print(f"label min: {label.min()} \t label max: {label.max()}")

# # plot images and labels
# plot_images_and_labels()

def main(args):

    # Setting the seed
    pl.seed_everything(args.seed)

    if args.dataset == 'zurich':
        dataset_root = "/home/GRAMES.POLYMTL.CA/u114716/datasets/sci-zurich_preprocessed_full_clean/"
    elif args.dataset == 'colorado':
        dataset_root = "/home/GRAMES.POLYMTL.CA/u114716/datasets/sci-colorado_preprocessed_clean/"
    # data_dir = "/home/GRAMES.POLYMTL.CA/u114716/datasets/sci-zurich_preprocessed_full_clean/"
    # data_dir = "/Users/nagakarthik/code/mri_datasets/sci-zurich_preprocessed_full/data_processed_clean/"

    save_path = args.save_path
    # save_dir = "/home/GRAMES.POLYMTL.CA/u114716/sci-zurich_project/modeling/saved_models"

    # define models
    # TODO: add options for more models
    if args.model in ["unet", "UNet"]:            
        net = UNet(
            spatial_dims=3, in_channels=1, out_channels=2,
            channels=(
                args.init_filters, 
                args.init_filters * 2, 
                args.init_filters * 4, 
                args.init_filters * 8, 
                args.init_filters * 16),
            strides=(2, 2, 2, 2),
            num_res_units=2,)
        exp_id =f"{args.dataset}_{args.model}_{args.loss_func}_{args.optimizer}_lr={args.learning_rate}"
    elif args.model in ["unetr", "UNETR"]:
        net = UNETR(
            in_channels=1, out_channels=2, 
            img_size=(64, 64, 128),
            feature_size=args.feature_size, 
            hidden_size=args.hidden_size, 
            mlp_dim=args.mlp_dim, 
            num_heads=args.num_heads,
            pos_embed="perceptron", 
            norm_name="instance", 
            res_block=True, dropout_rate=0.0,)
        exp_id =f"{args.dataset}_{args.model}_{args.loss_func}_{args.optimizer}_lr={args.learning_rate}" \
                f"_fs={args.feature_size}_hs={args.hidden_size}_mlpd={args.mlp_dim}_nh={args.num_heads}"
    elif args.model in ["segresnet", "SegResNet"]:
        net = SegResNet(
            in_channels=1, out_channels=2,
            init_filters=16,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            dropout_prob=0.2,)
        exp_id =f"{args.dataset}_{args.model}_{args.loss_func}_{args.optimizer}_lr={args.learning_rate}"

    # Define the loss function and the optimizer
    ce_weights = torch.FloatTensor([0.001, 0.999])  # for bg, fg
    if args.loss_func == "dice_ce":
        loss_function = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True, ce_weight=ce_weights)
    elif args.loss_func == "dice_ce_sq":
        loss_function = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True, ce_weight=ce_weights, squared_pred=True)
    elif args.loss_func == "dice":
        loss_function = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)    # because there are 2 classes (bg=0 and fg=1)
    else:
        loss_function = FocalLoss(include_background=False, to_onehot_y=True, gamma=2.0)
    # loss_function = MaskedDiceLoss(to_onehot_y=True, softmax=True)

    if args.optimizer in ["adamw", "AdamW", "Adamw"]:
        optimizer_class = torch.optim.AdamW
    elif args.optimizer in ["SGD", "sgd"]:
        optimizer_class = torch.optim.SGD

    # instantiate the PL model
    pl_model = Model(args, data_root=dataset_root, net=net, loss_function=loss_function, optimizer_class=optimizer_class)

    wandb_logger = pl.loggers.WandbLogger(
                        name=exp_id,
                        group=args.model, 
                        log_model=True, # save best model using checkpoint callback
                        project=f"sci-{args.dataset}",
                        entity='naga-karthik',
                        config=args)

    # to save the best model on validation
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_path, filename=exp_id,   
        monitor='val_dice', save_top_k=1, mode="max", save_last=False)
    
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_dice", min_delta=0.00,
        patience=args.patience, verbose=False, mode="max")

    if not args.only_eval:
        # initialise Lightning's trainer.
        trainer = pl.Trainer(
            devices=args.num_gpus, accelerator="gpu", strategy="ddp",
            logger=wandb_logger, 
            callbacks=[checkpoint_callback, lr_monitor, early_stopping],
            check_val_every_n_epoch=args.check_val_every_n_epochs,
            max_epochs=args.max_epochs, 
            precision=32,
            enable_progress_bar=args.enable_progress_bar)

        # Train!
        trainer.fit(pl_model)
        print("------- Training Done! -------")

        print("------- Printing the Best Model Path! ------")     # the PyTorch Lightning way
        # print best checkpoint after training
        print(trainer.checkpoint_callback.best_model_path)
    
    else:
        print("------- Loading the Best Model! ------")     # the PyTorch Lightning way
        # load the best checkpoint after training
        loaded_model = pl_model.load_from_checkpoint(os.path.join(args.save_path, exp_id)+ ".ckpt", strict=False)
        print("------- Testing Begins! -------")
        # TODO: a way to test the model


def get_ce_weights(label):    
    '''
    label/target: shape - [C, D, H, W]  
    '''
    if label.shape[0] > 1:
        label = label[1:]  # for One-Hot format data, remove the background channel
    
    label_flat = torch.ravel(torch.any(label,0))  # in case label has multiple dimensions
    num_fg_indices = torch.nonzero(label_flat).shape[0]
    num_bg_indices = torch.nonzero(~label_flat).shape[0]

    counts = torch.FloatTensor([num_bg_indices, num_fg_indices])
    if num_fg_indices == 0:
        counts[1] = 1e-5    # to prevent division by zero
        # and discard the loss coming only from the bg_indices
    
    ce_weights = 1.0/counts
    norm_ce_weights = ce_weights/ce_weights.sum()

    return norm_ce_weights.cuda()

def visualize(preds, imgs, gts, num_slices=10):
    # getting ready for post processing
    imgs, gts = imgs.detach().cpu(), gts.detach().cpu(), 
    imgs = imgs.squeeze(dim=1).numpy()  # shape: (1, 64, 64, -1)
    gts = gts.squeeze(dim=1)
    preds = torch.argmax(preds, dim=1).detach().cpu()

    fig, axs = plt.subplots(3, num_slices, figsize=(12, 3))
    fig.suptitle('Original --> Ground Truth --> Prediction')
    mid_sag_slice = imgs.shape[1]//2
    slice_nums = np.arange(-mid_sag_slice, mid_sag_slice+1)

    for i in range(num_slices):
        axs[0, i].imshow(imgs[0, slice_nums[i], :, :], cmap='gray'); axs[0, i].axis('off') 
        axs[1, i].imshow(gts[0, slice_nums[i], :, :]); axs[1, i].axis('off')    
        axs[2, i].imshow(preds[0, slice_nums[i], :, :]); # axs[2, i].axis('off')
    
    plt.tight_layout()
    fig.show()
    return fig



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script for training custom models for SCI Lesion Segmentation.')
    # Arguments for model, data, and training and saving
    parser.add_argument('-e', '--only_eval', default=False, action='store_true', help='Only do evaluation, i.e. skip training!')
    parser.add_argument('-m', '--model', 
                        choices=['unet', 'UNet', 'unetr', 'UNETR', 'segresnet', 'SegResNet'], 
                        default='unet', type=str, help='Model type to be used')
    # dataset
    parser.add_argument("--dataset", type=str, default='zurich', help="dataset to be used.")

    # unet model 
    # parser.add_argument('-t', '--task', choices=['sc', 'mc'], default='sc', type=str, help="Single-channel or Multi-channel model ")
    parser.add_argument('-initf', '--init_filters', default=16, type=int, help="Number of Filters in Init Layer")
    # parser.add_argument('-ccs', '--center_crop_size', nargs='+', default=[128, 256, 96], help='List containing center crop size for preprocessing')
    # parser.add_argument('-svs', '--subvolume_size', nargs='+', default=[128, 256, 96], help='List containing subvolume size')
    # parser.add_argument('-srs', '--stride_size', nargs='+', default=[128, 256, 96], help='List containing stride size')

    # unetr model 
    parser.add_argument('-fs', '--feature_size', default=16, type=int, help="Feature Size")
    parser.add_argument('-hs', '--hidden_size', default=768, type=int, help='Dimensionality of hidden embeddings')
    parser.add_argument('-mlpd', '--mlp_dim', default=2048, type=int, help='Dimensionality of MLP layer')
    parser.add_argument('-nh', '--num_heads', default=12, type=int, help='Number of heads in Multi-head Attention')
    
    # optimizations
    parser.add_argument('-lf', '--loss_func',
                         choices=['dice', 'dice_ce', 'dice_f'],
                         default='dice', type=str, help="Loss function to use")
    parser.add_argument('-gpus', '--num_gpus', default=1, type=int, help="Number of GPUs to use")
    parser.add_argument('-nw', '--num_workers', default=4, type=int, help='Number of workers for the dataloaders')
    parser.add_argument('-p', '--precision', default=32, type=int, help='Precision for training')
    parser.add_argument('-me', '--max_epochs', default=1000, type=int, help='Number of epochs for the training process')
    parser.add_argument('-bs', '--batch_size', default=2, type=int, help='Batch size of the training and validation processes')
    parser.add_argument('-opt', '--optimizer', 
                        choices=['adamw', 'AdamW', 'SGD', 'sgd'], 
                        default='adamw', type=str, help='Optimizer to use')
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help='Learning rate for training the model')
    parser.add_argument('-pat', '--patience', default=200, type=int, help='number of validation steps (val_every_n_iters) to wait before early stopping')
    parser.add_argument('--T_0', default=100, type=int, help='number of steps in each cosine cycle')
    parser.add_argument('-epb', '--enable_progress_bar', default=False, action='store_true', help='by default is disabled since it doesnt work in colab')
    parser.add_argument('-cve', '--check_val_every_n_epochs', default=30, type=int, help='num of epochs to wait before validation')
    # saving
    parser.add_argument('-mn', '--model_name', default='unet3d', type=str, help='Model ID to-be-used for saving the .pt saved model file')
    parser.add_argument('-sp', '--save_path', 
                        default=f"/home/GRAMES.POLYMTL.CA/u114716/sci-zurich_project/modeling/saved_models", 
                        type=str, help='Path to the saved models directory')
    parser.add_argument('-c', '--continue_from_checkpoint', default=False, action='store_true', help='Load model from checkpoint and continue training')
    parser.add_argument('-se', '--seed', default=42, type=int, help='Set seeds for reproducibility')
    parser.add_argument('-v', '--visualize_test_preds', default=False, action='store_true',
                        help='Enable to save subvolume predictions during the test phase for visual assessment')

    args = parser.parse_args()

    main(args)

    # # check best model output with input image and label 
    # model.load_state_dict(torch.load(os.path.join(save_dir, "best_metric_zurich_masked_model.pth")))
    # model.eval()
    # with torch.no_grad():
    #     for i, val_data in enumerate(val_loader):
    #         roi_size = voxel_size
    #         sw_batch_size = 4
    #         val_input = val_data["image"]
    #         val_outputs = sliding_window_inference(val_input.to(device), roi_size, sw_batch_size, model)
    #         val_outputs_fin = torch.argmax(val_outputs, dim=1).detach().cpu()
            
    #         # plot the mid slice
    #         plt.figure("check", (18, 6))
    #         plt.subplot(1, 3, 1)
    #         plt.title(f"image {i}")
    #         plt.imshow(val_input[0, 0, val_input.shape[2]//2, :, :] , cmap="gray")
    #         plt.subplot(1, 3, 2)
    #         plt.title(f"label {i}")
    #         plt.imshow(val_data["label"][0, 0, val_input.shape[2]//2, :, :])
    #         plt.subplot(1, 3, 3)
    #         plt.title(f"output {i}")
    #         plt.imshow(val_outputs_fin[0, val_input.shape[2]//2, :, :])
    #         # plt.show()
    #         plt.savefig(os.path.join(figs_dir, f"val_preds_{i}"))