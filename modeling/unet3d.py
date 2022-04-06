import copy
import math

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

def conv_norm_lrelu(feat_in, feat_out):
    """Conv3D + InstanceNorm3D + LeakyReLU block"""
    return nn.Sequential(
        nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
        nn.InstanceNorm3d(feat_out),
        nn.LeakyReLU()
    )


def norm_lrelu_conv(feat_in, feat_out):
    """InstanceNorm3D + LeakyReLU + Conv3D block"""
    return nn.Sequential(
        nn.InstanceNorm3d(feat_in),
        nn.LeakyReLU(),
        nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False)
    )


def lrelu_conv(feat_in, feat_out):
    """LeakyReLU + Conv3D block"""
    return nn.Sequential(
        nn.LeakyReLU(),
        nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False)
    )


def norm_lrelu_upscale_conv_norm_lrelu(feat_in, feat_out):
    """InstanceNorm3D + LeakyReLU + 2X Upsample + Conv3D + InstanceNorm3D + LeakyReLU block"""
    return nn.Sequential(
        nn.InstanceNorm3d(feat_in),
        nn.LeakyReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
        nn.InstanceNorm3d(feat_out),
        nn.LeakyReLU()
    )

# ---------------------------- ModifiedUNet3D Encoder Implementation -----------------------------
class ModifiedUNet3DEncoder(nn.Module):
    """Encoder for ModifiedUNet3D. Adapted from ivadomed.models"""
    def __init__(self, cfg, in_channels=1, base_n_filter=8, flatten=True, attention=False):
        super(ModifiedUNet3DEncoder, self).__init__()
        self.cfg = cfg

        self.flatten = flatten
        self.attention = attention

        # Initialize common operations
        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.5)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Level 1 context pathway
        self.conv3d_c1_1 = nn.Conv3d(in_channels, base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_c1_2 = nn.Conv3d(base_n_filter, base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu_conv_c1 = lrelu_conv(base_n_filter, base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(base_n_filter)

        # Level 2 context pathway
        self.conv3d_c2 = nn.Conv3d(base_n_filter, base_n_filter * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c2 = norm_lrelu_conv(base_n_filter * 2, base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(base_n_filter * 2)

        # Level 3 context pathway
        self.conv3d_c3 = nn.Conv3d(base_n_filter * 2, base_n_filter * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c3 = norm_lrelu_conv(base_n_filter * 4, base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(base_n_filter * 4)

        # Level 4 context pathway
        self.conv3d_c4 = nn.Conv3d(base_n_filter * 4, base_n_filter * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c4 = norm_lrelu_conv(base_n_filter * 8, base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(base_n_filter * 8)

        # Level 5 context pathway, level 0 localization pathway
        self.conv3d_c5 = nn.Conv3d(base_n_filter * 8, base_n_filter * 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c5 = norm_lrelu_conv(base_n_filter * 16, base_n_filter * 16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = norm_lrelu_upscale_conv_norm_lrelu(base_n_filter * 16, base_n_filter * 8)

        if self.flatten:
            self.fc = nn.Linear(self.cfg.unet_encoder_out_dim, self.cfg.unet_encoder_out_dim)

    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)

        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out

        # Level 4 context pathway
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out

        # Level 5
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5

        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

        context_features = [context_1, context_2, context_3, context_4]

        return out, context_features


# ---------------------------- ModifiedUNet3D Decoder Implementation -----------------------------
class ModifiedUNet3DDecoder(nn.Module):
    """Decoder for ModifiedUNet3D. Adapted from ivadomed.models"""
    def __init__(self, cfg, n_classes=1, base_n_filter=8):
        super(ModifiedUNet3DDecoder, self).__init__()
        self.cfg = cfg

        # Initialize common operations
        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.5)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv3d_l0 = nn.Conv3d(base_n_filter * 8, base_n_filter * 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(base_n_filter * 8)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = conv_norm_lrelu(base_n_filter * 16, base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(base_n_filter * 16, base_n_filter * 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = norm_lrelu_upscale_conv_norm_lrelu(base_n_filter * 8, base_n_filter * 4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = conv_norm_lrelu(base_n_filter * 8, base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(base_n_filter * 8, base_n_filter * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = norm_lrelu_upscale_conv_norm_lrelu(base_n_filter * 4, base_n_filter * 2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = conv_norm_lrelu(base_n_filter * 4, base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(base_n_filter * 4, base_n_filter * 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = norm_lrelu_upscale_conv_norm_lrelu(base_n_filter * 2, base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = conv_norm_lrelu(base_n_filter * 2, base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3d(base_n_filter * 2, n_classes, kernel_size=1, stride=1, padding=0, bias=False)

        self.ds2_1x1_conv3d = nn.Conv3d(base_n_filter * 8, n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(base_n_filter * 4, n_classes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, context_features):
        # Get context features from the encoder
        context_1, context_2, context_3, context_4 = context_features

        out = self.conv3d_l0(x)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)

        # Level 1 localization pathway
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

        # Level 2 localization pathway
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

        # Level 3 localization pathway
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

        # Level 4 localization pathway
        out = torch.cat([out, context_1], dim=1)
        out = self.conv_norm_lrelu_l4(out)
        out_pred = self.conv3d_l4(out)

        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds1_ds2_sum_upscale = self.upsample(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsample(ds1_ds2_sum_upscale_ds3_sum)

        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale

        # Final Activation Layer
        out = F.relu(out) / F.relu(out).max() if bool(F.relu(out).max()) else F.relu(out)
        out = out.squeeze()

        return out


# ---------------------------- ModifiedUNet3D Implementation -----------------------------
class ModifiedUNet3D(nn.Module):
    """ModifiedUNet3D with Encoder + Decoder. Adapted from ivadomed.models"""
    def __init__(self, cfg):
        super(ModifiedUNet3D, self).__init__()
        self.cfg = cfg
        self.unet_encoder = ModifiedUNet3DEncoder(cfg, in_channels=1 if cfg.task == 'sc' else 2,
                                                  base_n_filter=cfg.base_n_filter, flatten=False)
        self.unet_decoder = ModifiedUNet3DDecoder(cfg, n_classes=1, base_n_filter=cfg.base_n_filter)

    def forward(self, x1, x2=None):
        # x1: (B, 1, SV, SV, SV), x2: (B, 1, SV, SV, SV)

        if self.cfg.task == 'mc':   # mc: multi-"channel"
            # Concat. TPs (to be used when axial scan is also included)
            x = torch.cat([x1, x2], dim=1).to(x1.device)    # x: (B, 2, SV, SV, SV)
        else:
            # Discard x2
            x = x1      # x: (B, 1, SV, SV, SV)

        x, context_features = self.unet_encoder(x)
        # x: (B, 8 * F, SV // 8, SV // 8, SV // 8)
        # context_features: [4]
        #   0 -> (B, F, SV, SV, SV)
        #   1 -> (B, 2 * F, SV / 2, SV / 2, SV / 2)
        #   2 -> (B, 4 * F, SV / 4, SV / 4, SV / 4)
        #   3 -> (B, 8 * F, SV / 8, SV / 8, SV / 8)

        seg_logits = self.unet_decoder(x, context_features)

        return seg_logits
