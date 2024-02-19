# ------------------------------------------------------------------------------
# Written by Rongchang Xie (rongchangxie@pku.edu.cn) 
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from collections import OrderedDict
import numpy as np
import cv2
import random

from .pose_hrnet import PoseHighResolutionNet
from core.inference import get_max_preds_tensor

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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
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

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

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

        self.final_layer = nn.Conv2d(
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
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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
        x = self.layer4(x)

        fea = self.deconv_layers(x)
        ht = self.final_layer(fea)

        return fea,ht

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
            for m in self.final_layer.modules():
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

class PoseCons(nn.Module):

    def __init__(self, resnet, cfg, resnet_tch=None, resnet_seg=None, **kwargs):
        super(PoseCons, self).__init__()
        self.resnet = resnet

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.image_size = cfg.MODEL.IMAGE_SIZE

        self.cfg = cfg

    def get_batch_affine_transform(self, batch_size):

        sf = self.scale_factor
        rf = self.rotation_factor
        batch_trans  = []
        for b in range(batch_size):
            r = 0
            s = 1
            c = self.image_size/2
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                    if random.random() <= 0.8 else 0

            trans = cv2.getRotationMatrix2D((0,0), r, s)
            batch_trans.append(trans)

        batch_trans = np.stack(batch_trans, 0)
        batch_trans = torch.from_numpy(batch_trans).cuda()

        return batch_trans

    def forward(self, x, meta=None, aug_couple=["A30", "A60"]):
        if type(x) != list:
            return self.resnet(x)[1]
        
        # Augment type of [teacher_net, student_net]
        [teacher_aug, student_aug] = aug_couple
        
        # RandAug
        # if self.cfg.CONS_RAND_AUG:
        if self.cfg.USE_RandAug_AUG:  # can only do for PIL images (not tensors)
            sup_x, unsup_x, aug_unsup_x = x
        else:
            sup_x, unsup_x = x
        cons_x = unsup_x.clone()

        batch_size = sup_x.shape[0]
        _, sup_ht = self.resnet(sup_x)


        # Teacher
        with torch.no_grad():
            # _, unsup_ht = self.resnet(unsup_x)
            
            if teacher_aug == "A30":
                unsup_aug = unsup_x
            elif teacher_aug == "A60":
                theta = self.get_batch_affine_transform(sup_x.shape[0])
                grid = F.affine_grid(theta, unsup_x.size()).float()
                unsup_aug = F.grid_sample(unsup_x, grid)
            elif teacher_aug == "JC":  # Joints Cutout
                _, unsup_ht_tmp = self.resnet(unsup_x)
                preds, _ = get_max_preds_tensor(unsup_ht_tmp.detach())
                unsup_aug = mask_joint(unsup_x, preds*4, self.cfg.MASK_JOINT_NUM)
            elif teacher_aug == "JO":  #  Semi-supervised Cut-Occlude (SSCO)
                _, unsup_ht_tmp = self.resnet(unsup_x)
                preds, _ = get_max_preds_tensor(unsup_ht_tmp.detach())
                unsup_aug = keypoint_cutmix_perception(unsup_x, preds*4, self.cfg.KP_CutMix_JOINT_NUM)
            elif teacher_aug == "CO":
                unsup_aug = random_cutout(unsup_x, self.cfg.MASK_HOLES_NUM_CO)
            elif teacher_aug == "MU":
                unsup_aug = random_mixup(unsup_x)
            elif teacher_aug == "CM":
                unsup_aug = random_cutmix(unsup_x, self.cfg.MASK_HOLES_NUM_CM)
            elif teacher_aug == "RA":
                unsup_aug = aug_unsup_x.clone()

            _, unsup_ht = self.resnet(unsup_aug)
            
        
        with torch.no_grad():
            preds, _ = get_max_preds_tensor(unsup_ht.detach())
            
            if not self.cfg.MULTI_AUGS:
            
                if not self.cfg.REPEAT_AUG:
                    if "RA" in student_aug or self.cfg.USE_RandAug_AUG:
                        cons_x = aug_unsup_x.clone()
                
                    # Strong Augmentation #1, Joints Cutout 
                    if "JC" in student_aug or self.cfg.USE_JointCutout_AUG:
                        cons_x = mask_joint(cons_x, preds*4, self.cfg.MASK_JOINT_NUM)
                        
                    # Semi-supervised Cut-Occlude (SSCO)
                    if "JO" in student_aug or self.cfg.USE_JointCutMix_AUG:
                        cons_x = keypoint_cutmix_perception(cons_x, preds*4, self.cfg.KP_CutMix_JOINT_NUM)
                    
                    # Cutout, Mixup, CutMix, RandAugment
                    if "CO" in student_aug or self.cfg.USE_Cutout_AUG:
                        cons_x = random_cutout(cons_x, self.cfg.MASK_HOLES_NUM_CO)
                        
                    if "MU" in student_aug or self.cfg.USE_MixUp_AUG:
                        cons_x = random_mixup(cons_x)
                        
                    if "CM" in student_aug or self.cfg.USE_CutMix_AUG:
                        cons_x = random_cutmix(cons_x, self.cfg.MASK_HOLES_NUM_CM)

                else:
                    cons_x_list = []  # apply one single aug repeatly for multiple unsupervised losses
                    for _ in range(self.cfg.REPEAT_TIMES):  # only one kind of aug will be true
                        if self.cfg.USE_JointCutout_AUG:  # JC
                            cons_x_copy = cons_x.clone()
                            cons_x_list.append(mask_joint(cons_x_copy, preds*4, self.cfg.MASK_JOINT_NUM))
                        if self.cfg.USE_JointCutMix_AUG:  # JO
                            cons_x_copy = cons_x.clone()
                            cons_x_list.append(keypoint_cutmix_perception(cons_x_copy, preds*4, self.cfg.KP_CutMix_JOINT_NUM))
                        if self.cfg.USE_Cutout_AUG:  # CO
                            cons_x_copy = cons_x.clone()
                            cons_x_list.append(random_cutout(cons_x_copy, self.cfg.MASK_HOLES_NUM_CO))
                        if self.cfg.USE_MixUp_AUG:  # MU
                            cons_x_copy = cons_x.clone()
                            cons_x_list.append(random_mixup(cons_x_copy))  
                        if self.cfg.USE_CutMix_AUG:  # CM
                            cons_x_copy = cons_x.clone()
                            cons_x_list.append(random_cutmix(cons_x_copy, self.cfg.MASK_HOLES_NUM_CM))

                        if self.cfg.USE_JointCutMix_CutOut_AUG:  # JO+CO
                            cons_x_copy = cons_x.clone()
                            cons_x_copy = keypoint_cutmix_perception(cons_x_copy, preds*4, self.cfg.KP_CutMix_JOINT_NUM)
                            cons_x_copy = random_cutout(cons_x_copy, self.cfg.MASK_HOLES_NUM_CO)
                            cons_x_list.append(cons_x_copy)
                        if self.cfg.USE_JointCutout_CutMix_AUG:  # JC+CM
                            cons_x_copy = cons_x.clone()
                            cons_x_copy = mask_joint(cons_x_copy, preds*4, self.cfg.MASK_JOINT_NUM)
                            cons_x_copy = random_cutmix(cons_x_copy, self.cfg.MASK_HOLES_NUM_CM)
                            cons_x_list.append(cons_x_copy)

            else:  # self.cfg.MULTI_AUGS and self.cfg.REPEAT_AUG will not be True in the same time.
                cons_x_list = []  # apply multiple separate augs for multiple unsupervised losses
                if self.cfg.USE_JointCutout_AUG:  # JC
                    cons_x_copy = cons_x.clone()
                    cons_x_list.append(mask_joint(cons_x_copy, preds*4, self.cfg.MASK_JOINT_NUM))
                if self.cfg.USE_JointCutMix_AUG:  # JO
                    cons_x_copy = cons_x.clone()
                    cons_x_list.append(keypoint_cutmix_perception(cons_x_copy, preds*4, self.cfg.KP_CutMix_JOINT_NUM))
                if self.cfg.USE_Cutout_AUG:  # CO
                    cons_x_copy = cons_x.clone()
                    cons_x_list.append(random_cutout(cons_x_copy, self.cfg.MASK_HOLES_NUM_CO))
                if self.cfg.USE_MixUp_AUG:  # MU
                    cons_x_copy = cons_x.clone()
                    cons_x_list.append(random_mixup(cons_x_copy))  
                if self.cfg.USE_CutMix_AUG:  # CM
                    cons_x_copy = cons_x.clone()
                    cons_x_list.append(random_cutmix(cons_x_copy, self.cfg.MASK_HOLES_NUM_CM))

                if self.cfg.USE_RandAug_AUG:  # RA
                    cons_x_list.append(aug_unsup_x.clone())

                if self.cfg.USE_JointCutMix_CutOut_AUG:  # JO+CO
                    cons_x_copy = cons_x.clone()
                    cons_x_copy = keypoint_cutmix_perception(cons_x_copy, preds*4, self.cfg.KP_CutMix_JOINT_NUM)
                    cons_x_copy = random_cutout(cons_x_copy, self.cfg.MASK_HOLES_NUM_CO)
                    cons_x_list.append(cons_x_copy)
                if self.cfg.USE_JointCutout_CutMix_AUG:  # JC+CM
                    cons_x_copy = cons_x.clone()
                    cons_x_copy = mask_joint(cons_x_copy, preds*4, self.cfg.MASK_JOINT_NUM)
                    cons_x_copy = random_cutmix(cons_x_copy, self.cfg.MASK_HOLES_NUM_CM)
                    cons_x_list.append(cons_x_copy)


        # Transform
        # Apply Affine Transformation again for hard augmentation
        if self.cfg.UNSUP_TRANSFORM:
            theta = self.get_batch_affine_transform(sup_x.shape[0])
            grid = F.affine_grid(theta, cons_x.size()).float()
            if (not self.cfg.MULTI_AUGS) and (not self.cfg.REPEAT_AUG):
                cons_x = F.grid_sample(cons_x, grid)
            else:
                cons_x_list = [F.grid_sample(cons_x, grid) for cons_x in cons_x_list]

            if teacher_aug == "A60":
                unsup_ht_trans = unsup_ht.detach()
            else:
                ht_grid = F.affine_grid(theta, unsup_ht.size()).float()
                unsup_ht_trans = F.grid_sample(unsup_ht, ht_grid)
        else:
            unsup_ht_trans = unsup_ht.detach()
            theta = None

        # Student
        if (not self.cfg.MULTI_AUGS) and (not self.cfg.REPEAT_AUG):
            _, cons_ht = self.resnet(cons_x)
            
            # return [sup_ht, unsup_ht, unsup_ht_trans, cons_x, cons_ht, None]
            return [sup_ht, unsup_ht, unsup_ht_trans, cons_x, cons_ht, theta]

        else:
            cons_ht_list = [self.resnet(cons_x)[-1] for cons_x in cons_x_list]
            
            # return [sup_ht, unsup_ht, unsup_ht_trans, cons_x_list, cons_ht_list, None]
            return [sup_ht, unsup_ht, unsup_ht_trans, cons_x_list, cons_ht_list, theta]



def get_pose_net(cfg, is_train, **kwargs):

    if cfg.MODEL.BACKBONE == 'resnet':
        num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
        style = cfg.MODEL.STYLE
        block_class, layers = resnet_spec[num_layers]

        if style == 'caffe':
            block_class = Bottleneck_CAFFE
        resnet = PoseResNet(block_class, layers, cfg, **kwargs)
    elif cfg.MODEL.BACKBONE == 'hrnet':
        resnet = PoseHighResolutionNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        resnet.init_weights(cfg.MODEL.PRETRAINED)

    model = PoseCons(resnet, cfg)

    return model
