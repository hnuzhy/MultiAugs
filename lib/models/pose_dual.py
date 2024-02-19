# ------------------------------------------------------------------------------
# Written by Rongchang Xie (rongchangxie@pku.edu.cn) 
# Re-written by Huayi Zhou (sjtu_zhy@sjtu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import contextlib
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch.autograd import Function
from collections import OrderedDict
import numpy as np
import random
import cv2
from utils.transforms import get_affine_transform, cvt_MToTheta
from core.inference import get_max_preds_tensor
from core.loss import JointsMSELoss

from .pose_hrnet import PoseHighResolutionNet

from utils.augments import random_cutout
from utils.augments import random_mixup
from utils.augments import random_cutmix
from utils.augments import mask_joint
from utils.augments import keypoint_cutmix_perception


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(Bottleneck, self).__init__()

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups = groups, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(width, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck_CAFFE(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_CAFFE, self).__init__()
        # add stride to conv1x1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):

    def __init__(self, block, layers, cfg, groups=1, width_per_group=64, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        self.cfg = cfg
        self.groups = groups
        self.base_width = width_per_group

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer_1 = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups = self.groups,
                           base_width = self.base_width))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        fea = self.layer4(x)

        x = self.deconv_layers(fea)
        ht_1 = self.final_layer_1(x)

        return fea, ht_1

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer_1.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            logger.info('=> loading pretrained model {}'.format(pretrained))
            checkpoint = torch.load(pretrained, map_location = 'cpu')
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict_old = checkpoint['state_dict']
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith('module.'):
                        state_dict[key[7:]] = state_dict_old[key]
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(pretrained))
                    
            # delete 'module.' 
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]:v for k,v in state_dict.items()}

            # if list(state_dict.keys())[0][:6] == 'resnet2':
            #     state_dict = {k[8:]:v for k,v in state_dict.items()}

            if list(state_dict.keys())[0][:6] == 'resnet':
                state_dict = {k[7:]:v for k,v in state_dict.items()}

            self.load_state_dict(state_dict, strict=False)
        else:
            logger.error('=> imagenet pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


class PoseCotrain(nn.Module):

    def __init__(self, resnet, resnet2, cfg, resnet_tch = None, resnet_seg = None , **kwargs):
        super(PoseCotrain, self).__init__()
        # np.random.seed(1314)
        self.resnet = resnet
        self.resnet2 = resnet2

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.cfg = cfg

        self.multi_infer = False
        self.flip = False
        self.scale_set = [0.7,0.9,1.1,1.3]
        self.flip_pairs = [[1,4],[2,5],[3,6],[14,17],[15,18],[16,19],[20,21],[22,23]]

    def get_batch_affine_transform(self, batch_size):
        sf = self.scale_factor
        rf = self.rotation_factor
        # shift_f = 0.1
        # shear_f = 0.10

        batch_trans  = []
        for b in range(batch_size):
            r,s = 1,1
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                    if random.random() <= 0.8 else 0
            trans = cv2.getRotationMatrix2D((0,0), r, s)

            batch_trans.append(trans)
        batch_trans = np.stack(batch_trans, 0)
        batch_trans = torch.from_numpy(batch_trans).cuda()
        return batch_trans


    def forward(self, x, meta=None, aug_couple=["A30", "A60"]):
        # Training
        if type(x)==list:
        
            # Augment type of [teacher_net, student_net]
            [teacher_aug, student_aug] = aug_couple
        
            # RandAug
            # if self.cfg.CONS_RAND_AUG:
            if self.cfg.USE_RandAug_AUG:  # can only do for PIL images (not tensors)
                sup_x, unsup_x, aug_unsup_x = x
            else:
                sup_x, unsup_x = x

            batch_size = x[0].shape[0]
            sup_fea1, sup_ht1 = self.resnet(sup_x)
            sup_fea2, sup_ht2 = self.resnet2(sup_x)


            # Teacher
            # Easy Augmentation
            with torch.no_grad():
                # unsup_fea1, unsup_ht1 = self.resnet(unsup_x)
                # unsup_fea2, unsup_ht2 = self.resnet2(unsup_x)
                
                unsup_x_1 = unsup_x.clone()
                unsup_x_2 = unsup_x.clone()
                
                if teacher_aug == "A30":
                    unsup_aug_1 = unsup_x_1
                    unsup_aug_2 = unsup_x_2
                elif teacher_aug == "A60":
                    theta = self.get_batch_affine_transform(sup_x.shape[0])
                    grid = F.affine_grid(theta, unsup_x_1.size()).float()
                    unsup_aug_1 = F.grid_sample(unsup_x_1, grid)
                    theta = self.get_batch_affine_transform(sup_x.shape[0])
                    grid = F.affine_grid(theta, unsup_x_2.size()).float()
                    unsup_aug_2 = F.grid_sample(unsup_x_2, grid)
                elif teacher_aug == "JC":  # Joints Cutout
                    _, unsup_ht1_tmp = self.resnet(unsup_x_1)
                    _, unsup_ht2_tmp = self.resnet2(unsup_x_2)
                    preds_1, _ = get_max_preds_tensor(unsup_ht1_tmp.detach())
                    preds_2, _ = get_max_preds_tensor(unsup_ht2_tmp.detach())
                    unsup_aug_1 = mask_joint(unsup_x_1, preds_2*4, self.cfg.MASK_JOINT_NUM)
                    unsup_aug_2 = mask_joint(unsup_x_2, preds_1*4, self.cfg.MASK_JOINT_NUM)
                elif teacher_aug == "JO":  #  Semi-supervised Cut-Occlude (SSCO)
                    _, unsup_ht1_tmp = self.resnet(unsup_x_1)
                    _, unsup_ht2_tmp = self.resnet2(unsup_x_2)
                    preds_1, _ = get_max_preds_tensor(unsup_ht1_tmp.detach())
                    preds_2, _ = get_max_preds_tensor(unsup_ht2_tmp.detach())
                    unsup_aug_1 = keypoint_cutmix_perception(unsup_x_1, preds_2*4, self.cfg.KP_CutMix_JOINT_NUM)
                    unsup_aug_2 = keypoint_cutmix_perception(unsup_x_2, preds_1*4, self.cfg.KP_CutMix_JOINT_NUM)
                elif teacher_aug == "CO":
                    unsup_aug_1 = random_cutout(unsup_x_1, self.cfg.MASK_HOLES_NUM_CO)
                    unsup_aug_2 = random_cutout(unsup_x_2, self.cfg.MASK_HOLES_NUM_CO)
                elif teacher_aug == "MU":
                    unsup_aug_1 = random_mixup(unsup_x_1)
                    unsup_aug_2 = random_mixup(unsup_x_2)
                elif teacher_aug == "CM":
                    unsup_aug_1 = random_cutmix(unsup_x_1, self.cfg.MASK_HOLES_NUM_CM)
                    unsup_aug_2 = random_cutmix(unsup_x_2, self.cfg.MASK_HOLES_NUM_CM)
                    
                unsup_fea1, unsup_ht1 = self.resnet(unsup_aug_1)
                unsup_fea2, unsup_ht2 = self.resnet2(unsup_aug_2)

            unsup_x_trans = unsup_x.clone()
            unsup_x_trans_2 = unsup_x.clone()

            with torch.no_grad():
                preds_1, _ = get_max_preds_tensor(unsup_ht1.detach())
                preds_2, _ = get_max_preds_tensor(unsup_ht2.detach())
            
                if not self.cfg.MULTI_AUGS:
                    if not self.cfg.REPEAT_AUG:
                        if "RA" in student_aug or self.cfg.USE_RandAug_AUG:
                            unsup_x_trans = aug_unsup_x.clone()
                            unsup_x_trans_2 = aug_unsup_x.clone()
                    
                        # Joint Cutout, Masking joints
                        if "JC" in student_aug or self.cfg.USE_JointCutout_AUG:
                            unsup_x_trans = mask_joint(unsup_x_trans, preds_2*4, self.cfg.MASK_JOINT_NUM)
                            unsup_x_trans_2 = mask_joint(unsup_x_trans_2, preds_1*4, self.cfg.MASK_JOINT_NUM)
                        
                        # Semi-supervised Cut-Occlude (SSCO)
                        if "JO" in student_aug or self.cfg.USE_JointCutMix_AUG:
                            unsup_x_trans = keypoint_cutmix_perception(unsup_x_trans, preds_2*4, self.cfg.KP_CutMix_JOINT_NUM)
                            unsup_x_trans_2 = keypoint_cutmix_perception(unsup_x_trans_2, preds_1*4, self.cfg.KP_CutMix_JOINT_NUM)
                        
                        # Cutout, Mixup, CutMix
                        if "CO" in student_aug or self.cfg.USE_Cutout_AUG:
                            unsup_x_trans = random_cutout(unsup_x_trans, self.cfg.MASK_HOLES_NUM_CO)
                            unsup_x_trans_2 = random_cutout(unsup_x_trans_2, self.cfg.MASK_HOLES_NUM_CO)
                            
                        if "MU" in student_aug or self.cfg.USE_MixUp_AUG:
                            unsup_x_trans = random_mixup(unsup_x_trans)
                            unsup_x_trans_2 = random_mixup(unsup_x_trans_2) 
                            
                        if "CM" in student_aug or self.cfg.USE_CutMix_AUG:
                            unsup_x_trans = random_cutmix(unsup_x_trans, self.cfg.MASK_HOLES_NUM_CM)
                            unsup_x_trans_2 = random_cutmix(unsup_x_trans_2, self.cfg.MASK_HOLES_NUM_CM)
                        
                    else:
                        cons_x1_list, cons_x2_list = [], []  # apply one single aug repeatly for multiple unsupervised losses
                        for _ in range(self.cfg.REPEAT_TIMES):  # only one kind of aug will be true
                            if self.cfg.USE_JointCutout_AUG:
                                cons_x1_copy, cons_x2_copy = unsup_x_trans.clone(), unsup_x_trans_2.clone()
                                cons_x1_list.append(mask_joint(cons_x1_copy, preds_2*4, self.cfg.MASK_JOINT_NUM))
                                cons_x2_list.append(mask_joint(cons_x2_copy, preds_1*4, self.cfg.MASK_JOINT_NUM))
                            if self.cfg.USE_JointCutMix_AUG:
                                cons_x1_copy, cons_x2_copy = unsup_x_trans.clone(), unsup_x_trans_2.clone()
                                cons_x1_copy = keypoint_cutmix_perception(cons_x1_copy, preds_2*4, self.cfg.KP_CutMix_JOINT_NUM)
                                cons_x2_copy = keypoint_cutmix_perception(cons_x2_copy, preds_1*4, self.cfg.KP_CutMix_JOINT_NUM)
                                cons_x1_list.append(cons_x1_copy)
                                cons_x2_list.append(cons_x2_copy)
                            if self.cfg.USE_Cutout_AUG:
                                cons_x1_copy, cons_x2_copy = unsup_x_trans.clone(), unsup_x_trans_2.clone()
                                cons_x1_list.append(random_cutout(cons_x1_copy))
                                cons_x2_list.append(random_cutout(cons_x2_copy))
                            if self.cfg.USE_MixUp_AUG:
                                cons_x1_copy, cons_x2_copy = unsup_x_trans.clone(), unsup_x_trans_2.clone()
                                cons_x1_list.append(random_mixup(cons_x1_copy))
                                cons_x2_list.append(random_mixup(cons_x2_copy))
                            if self.cfg.USE_CutMix_AUG:
                                cons_x1_copy, cons_x2_copy = unsup_x_trans.clone(), unsup_x_trans_2.clone()
                                cons_x1_list.append(random_cutmix(cons_x1_copy))
                                cons_x2_list.append(random_cutmix(cons_x2_copy))
                            
                            if self.cfg.USE_JointCutMix_CutOut_AUG:  # JO+CO
                                cons_x1_copy, cons_x2_copy = unsup_x_trans.clone(), unsup_x_trans_2.clone()
                                cons_x1_copy = keypoint_cutmix_perception(cons_x1_copy, preds_2*4, self.cfg.KP_CutMix_JOINT_NUM)
                                cons_x2_copy = keypoint_cutmix_perception(cons_x2_copy, preds_1*4, self.cfg.KP_CutMix_JOINT_NUM)
                                cons_x1_copy = random_cutout(cons_x1_copy, self.cfg.MASK_HOLES_NUM_CO)
                                cons_x2_copy = random_cutout(cons_x2_copy, self.cfg.MASK_HOLES_NUM_CO)
                                cons_x1_list.append(cons_x1_copy)
                                cons_x2_list.append(cons_x2_copy)
                            if self.cfg.USE_JointCutout_CutMix_AUG:  # JC+CM
                                cons_x1_copy, cons_x2_copy = unsup_x_trans.clone(), unsup_x_trans_2.clone()
                                cons_x1_copy = mask_joint(cons_x1_copy, preds_2*4, self.cfg.MASK_JOINT_NUM)
                                cons_x2_copy = mask_joint(cons_x2_copy, preds_1*4, self.cfg.MASK_JOINT_NUM)
                                cons_x1_copy = random_cutmix(cons_x1_copy, self.cfg.MASK_HOLES_NUM_CO)
                                cons_x2_copy = random_cutmix(cons_x2_copy, self.cfg.MASK_HOLES_NUM_CO)
                                cons_x1_list.append(cons_x1_copy)
                                cons_x2_list.append(cons_x2_copy)

                else:
                    cons_x1_list, cons_x2_list = [], []  # apply multiple separate augs for multiple unsupervised losses
                    if self.cfg.USE_JointCutout_AUG:
                        cons_x1_copy, cons_x2_copy = unsup_x_trans.clone(), unsup_x_trans_2.clone()
                        cons_x1_list.append(mask_joint(cons_x1_copy, preds_2*4, self.cfg.MASK_JOINT_NUM))
                        cons_x2_list.append(mask_joint(cons_x2_copy, preds_1*4, self.cfg.MASK_JOINT_NUM))
                    if self.cfg.USE_JointCutMix_AUG:
                        cons_x1_copy, cons_x2_copy = unsup_x_trans.clone(), unsup_x_trans_2.clone()
                        cons_x1_copy = keypoint_cutmix_perception(cons_x1_copy, preds_2*4, self.cfg.KP_CutMix_JOINT_NUM)
                        cons_x2_copy = keypoint_cutmix_perception(cons_x2_copy, preds_1*4, self.cfg.KP_CutMix_JOINT_NUM)
                        cons_x1_list.append(cons_x1_copy)
                        cons_x2_list.append(cons_x2_copy)
                    if self.cfg.USE_Cutout_AUG:
                        cons_x1_copy, cons_x2_copy = unsup_x_trans.clone(), unsup_x_trans_2.clone()
                        cons_x1_list.append(random_cutout(cons_x1_copy))
                        cons_x2_list.append(random_cutout(cons_x2_copy))
                    if self.cfg.USE_MixUp_AUG:
                        cons_x1_copy, cons_x2_copy = unsup_x_trans.clone(), unsup_x_trans_2.clone()
                        cons_x1_list.append(random_mixup(cons_x1_copy))
                        cons_x2_list.append(random_mixup(cons_x2_copy))
                    if self.cfg.USE_CutMix_AUG:
                        cons_x1_copy, cons_x2_copy = unsup_x_trans.clone(), unsup_x_trans_2.clone()
                        cons_x1_list.append(random_cutmix(cons_x1_copy))
                        cons_x2_list.append(random_cutmix(cons_x2_copy))
                    
                    if self.cfg.USE_JointCutMix_CutOut_AUG:  # JO+CO
                        cons_x1_copy, cons_x2_copy = unsup_x_trans.clone(), unsup_x_trans_2.clone()
                        cons_x1_copy = keypoint_cutmix_perception(cons_x1_copy, preds_2*4, self.cfg.KP_CutMix_JOINT_NUM)
                        cons_x2_copy = keypoint_cutmix_perception(cons_x2_copy, preds_1*4, self.cfg.KP_CutMix_JOINT_NUM)
                        cons_x1_copy = random_cutout(cons_x1_copy, self.cfg.MASK_HOLES_NUM_CO)
                        cons_x2_copy = random_cutout(cons_x2_copy, self.cfg.MASK_HOLES_NUM_CO)
                        cons_x1_list.append(cons_x1_copy)
                        cons_x2_list.append(cons_x2_copy)
                    if self.cfg.USE_JointCutout_CutMix_AUG:  # JC+CM
                        cons_x1_copy, cons_x2_copy = unsup_x_trans.clone(), unsup_x_trans_2.clone()
                        cons_x1_copy = mask_joint(cons_x1_copy, preds_2*4, self.cfg.MASK_JOINT_NUM)
                        cons_x2_copy = mask_joint(cons_x2_copy, preds_1*4, self.cfg.MASK_JOINT_NUM)
                        cons_x1_copy = random_cutmix(cons_x1_copy, self.cfg.MASK_HOLES_NUM_CO)
                        cons_x2_copy = random_cutmix(cons_x2_copy, self.cfg.MASK_HOLES_NUM_CO)
                        cons_x1_list.append(cons_x1_copy)
                        cons_x2_list.append(cons_x2_copy)


            # Transform
            # Apply Affine Transformation again for hard augmentation
            if self.cfg.UNSUP_TRANSFORM:
                with torch.no_grad():
                    theta = self.get_batch_affine_transform(batch_size)
                    grid = F.affine_grid(theta, sup_x.size()).float()

                    if (not self.cfg.MULTI_AUGS) and (not self.cfg.REPEAT_AUG):
                        unsup_x_trans = F.grid_sample(unsup_x_trans, grid)
                        unsup_x_trans_2 = F.grid_sample(unsup_x_trans_2, grid)
                    else:
                        cons_x1_list = [F.grid_sample(cons_x, grid) for cons_x in cons_x1_list]
                        cons_x2_list = [F.grid_sample(cons_x, grid) for cons_x in cons_x2_list]

                    if teacher_aug == "A60":
                        unsup_ht_trans1 = unsup_ht1.detach().clone()
                        unsup_ht_trans2 = unsup_ht2.detach().clone()
                    else:
                        ht_grid = F.affine_grid(theta, unsup_ht1.size()).float()
                        unsup_ht_trans1 = F.grid_sample(unsup_ht1.detach(), ht_grid)
                        unsup_ht_trans2 = F.grid_sample(unsup_ht2.detach(), ht_grid)
            else:
                # Raw image
                theta = torch.eye(2, 3).repeat(batch_size, 1, 1).double().cuda()

                unsup_ht_trans1 = unsup_ht1.detach().clone()
                unsup_ht_trans2 = unsup_ht2.detach().clone()
                
            # Student
            # Hard Augmentation
            if (not self.cfg.MULTI_AUGS) and (not self.cfg.REPEAT_AUG):
                _, cons_ht1 = self.resnet(unsup_x_trans)
                _, cons_ht2 = self.resnet2(unsup_x_trans_2)
                
                out_dic = {  # not used
                    'unsup_x_trans':unsup_x_trans,
                    'unsup_x_trans_2': unsup_x_trans_2,
                    'theta':        theta,
                    }

                return sup_ht1, sup_ht2, unsup_ht1, unsup_ht2, unsup_ht_trans1, unsup_ht_trans2, cons_ht1, cons_ht2, out_dic
                
            else:
                cons_ht1_list = [self.resnet(cons_x)[-1] for cons_x in cons_x1_list]
                cons_ht2_list = [self.resnet2(cons_x)[-1] for cons_x in cons_x2_list]
                
                out_dic = {  # not used
                    'cons_x1_list': cons_x1_list,
                    'cons_x2_list': cons_x2_list,
                    'theta':        theta,
                    }
                    
                return sup_ht1, sup_ht2, unsup_ht1, unsup_ht2, unsup_ht_trans1, unsup_ht_trans2, cons_ht1_list, cons_ht2_list, out_dic

        # Inference     
        else:
            batch_size, _, height, width = x.shape

            fea1, ht1 = self.resnet(x)
            fea2, ht2 = self.resnet2(x)

            return ht1, ht2 
    

def get_pose_net(cfg, is_train, **kwargs):

    if cfg.MODEL.BACKBONE == 'resnet':
        num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
        style = cfg.MODEL.STYLE
        block_class, layers = resnet_spec[num_layers]
        if style == 'caffe':
            block_class = Bottleneck_CAFFE

        resnet = PoseResNet(block_class, layers, cfg, **kwargs)
        resnet2 = PoseResNet(block_class, layers, cfg, **kwargs)

    elif cfg.MODEL.BACKBONE == 'hrnet':
        resnet = PoseHighResolutionNet(cfg, **kwargs)
        resnet2 = PoseHighResolutionNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        resnet.init_weights(cfg.MODEL.PRETRAINED)

    if is_train and cfg.MODEL.INIT_WEIGHTS:

        logger.info('Model 2 => loading pretrained model {}'.format(cfg.MODEL.PRETRAINED))
        resnet2.init_weights(cfg.MODEL.PRETRAINED)
        
    model = PoseCotrain(resnet, resnet2, cfg)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        state_dict = torch.load(cfg.MODEL.PRETRAINED, map_location = 'cpu')
        if 'resnet2.conv1.weight' in state_dict:
            print('pretrained')
            model.load_state_dict(state_dict, strict=False)


    return model
