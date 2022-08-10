from abc import ABC, abstractmethod
import logging
import os
import shutil
import tempfile
import argparse

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import wandb
import torch

from monai.apps import CrossValidation
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss, DiceLoss, MaskedDiceLoss, FocalLoss
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.networks.nets import UNet, DynUNet, BasicUNet, SegResNet, UNETR
from monai.data import (DataLoader, CacheDataset, load_decathlon_datalist, decollate_batch, list_data_collate)
from monai.transforms import (AsDiscrete, AddChanneld, Compose, CropForegroundd, LoadImaged, Orientationd, RandFlipd, 
                    RandCropByPosNegLabeld, RandShiftIntensityd, ScaleIntensityRanged, Spacingd, RandRotate90d, ToTensord,
                    SpatialPadd, NormalizeIntensityd, EnsureType, RandScaleIntensityd, RandWeightedCropd, EnsureChannelFirstd,
                    AsDiscreted, RandSpatialCropSamplesd, HistogramNormalized, RandomBiasFieldd)
from monai.engines import (EnsembleEvaluator, SupervisedEvaluator, SupervisedTrainer)
from monai.handlers import MeanDice, StatsHandler, ValidationHandler, from_engine


