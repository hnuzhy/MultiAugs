# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Rongchang Xie (rongchangxie@pku.edu.cn) 
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from core.inference import get_max_preds, get_max_preds_tensor


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight, cfg):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight
        self.cfg = cfg

    def forward(self, output, target, target_weight=None, meta=None, unsup_flag=False):
        batch_size = output.size(0)
        num_joints = output.size(1)
        height = output.size(2)
        width = output.size(3)
        
        if self.cfg.CONF_MASKING and unsup_flag:
            preds, maxvals = get_max_preds_tensor(target.clone())  # unsup_ht or unsup_ht_trans1 or unsup_ht_trans2
            preds_mask = torch.greater(maxvals, self.cfg.CONF_THRESH).float()  # shape (batch_size, num_joints)

        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0
          
        '''
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        '''
        
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()  # shape (batch_size, img_w*img_h)
            heatmap_gt = heatmaps_gt[idx].squeeze()  # shape (batch_size, img_w*img_h)
            if self.use_target_weight:
                hm_pd = heatmap_pred.mul(target_weight[:, idx])
                hm_gt = heatmap_gt.mul(target_weight[:, idx])
            else:
                hm_pd, hm_gt = heatmap_pred, heatmap_gt
                
            if self.cfg.SHARPEN_PRED and unsup_flag:
                hm_pd = F.softmax(hm_pd, dim=1)
                hm_gt = F.softmax(hm_gt, dim=1)
            
            if self.cfg.CONF_MASKING and unsup_flag:
                # print(hm_pd.shape, hm_gt.shape, preds_mask.shape, target_weight.shape)
                hm_pd = hm_pd.mul(torch.unsqueeze(preds_mask[:, idx], 1))
                hm_gt = hm_gt.mul(torch.unsqueeze(preds_mask[:, idx], 1))
                
            loss += 0.5 * self.criterion(hm_pd, hm_gt)
        
        return loss / num_joints


class PoseCoLoss(nn.Module):
    def __init__(self, use_target_weight, cfg):
        super(PoseCoLoss, self).__init__()
        self.mse_criterion = JointsMSELoss(use_target_weight, cfg)

        self.use_target_weight = use_target_weight
        
        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.target_type = cfg.MODEL.EXTRA.TARGET_TYPE
        self.heatmap_size = cfg.MODEL.EXTRA.HEATMAP_SIZE
        self.sigma = cfg.MODEL.EXTRA.SIGMA
        self.num_joints = 24
        self.target_type = 'gaussian'

        self.cfg = cfg

    def forward(self, output, target, target_weight, meta):
        if type(target)==list:
            sup_target, unsup_target = target
            sup_target_weight, unsup_target_weight = target_weight
            sup_meta, upsup_meta = meta
        else:
            sup_target = target
            sup_target_weight = target_weight
            sup_meta = meta

        batch_size, joint_num, ht_height, ht_width = sup_target.shape
        pseudo_target = 0
        
        # if not self.cfg.MULTI_AUGS:
        if (not self.cfg.MULTI_AUGS) and (not self.cfg.REPEAT_AUG):
            sup_ht1, sup_ht2, unsup_ht1, unsup_ht2, unsup_ht_trans1, unsup_ht_trans2, cons_ht1, cons_ht2, out_dic = output
        else:
            sup_ht1, sup_ht2, unsup_ht1, unsup_ht2, unsup_ht_trans1, unsup_ht_trans2, cons_ht1_list, cons_ht2_list, out_dic = output

        batch_size = sup_ht1.size(0)
        num_joints = sup_ht1.size(1)   

        loss_pose = 0.5*self.mse_criterion(sup_ht1, sup_target, sup_target_weight)
        loss_pose += 0.5*self.mse_criterion(sup_ht2, sup_target, sup_target_weight)
        
        # if not self.cfg.MULTI_AUGS:
        if (not self.cfg.MULTI_AUGS) and (not self.cfg.REPEAT_AUG):
            loss_cons = self.mse_criterion(cons_ht1, unsup_ht_trans2.detach(), unsup_target_weight, unsup_flag=True)
            loss_cons += self.mse_criterion(cons_ht2, unsup_ht_trans1.detach(),  unsup_target_weight, unsup_flag=True)
        else:
            loss_cons = 0
            if self.cfg.HEAT_FUSION:
                cons_ht1_avg = torch.stack(cons_ht1_list, dim=0).sum(0) / len(cons_ht1_list)
                loss_cons += self.mse_criterion(cons_ht1_avg, unsup_ht_trans2.detach(), unsup_target_weight, unsup_flag=True)
                cons_ht2_avg = torch.stack(cons_ht2_list, dim=0).sum(0) / len(cons_ht2_list)
                loss_cons += self.mse_criterion(cons_ht2_avg, unsup_ht_trans1.detach(), unsup_target_weight, unsup_flag=True)
            else:
                for cons_ht in cons_ht1_list:
                    loss_cons += self.mse_criterion(cons_ht, unsup_ht_trans2.detach(), unsup_target_weight, unsup_flag=True)
                for cons_ht in cons_ht2_list:
                    loss_cons += self.mse_criterion(cons_ht, unsup_ht_trans1.detach(), unsup_target_weight, unsup_flag=True)
                
        pseudo_target = [unsup_ht_trans2.detach().cpu(), unsup_ht_trans1.detach().cpu()]

        loss = loss_pose + loss_cons 
        loss_dic = {
            'loss_pose': loss_pose,
            'loss_cons': loss_cons,
        }

        return loss, loss_dic, pseudo_target

class PoseDisLoss(nn.Module):
    def __init__(self, use_target_weight, cfg=None):
        super(PoseDisLoss, self).__init__()
        self.mse_criterion = JointsMSELoss(use_target_weight,cfg)

        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.target_type = cfg.MODEL.EXTRA.TARGET_TYPE
        self.heatmap_size = cfg.MODEL.EXTRA.HEATMAP_SIZE
        self.sigma = cfg.MODEL.EXTRA.SIGMA
        self.num_joints = 24
        self.target_type = 'gaussian'
        
        self.cfg = cfg


    def forward(self, output, target, target_weight, meta):
        # unpackage
        # if not self.cfg.MULTI_AUGS:
        if (not self.cfg.MULTI_AUGS) and (not self.cfg.REPEAT_AUG):
            sup_ht, _, unsup_ht, _, cons_ht, _ = output
        else:
            sup_ht, _, unsup_ht, _, cons_ht_list, _ = output

        batch_size, joint_num, ht_height, ht_width = sup_ht.shape

        sup_target, unsup_target = target
        sup_target_weight, unsup_target_weight = target_weight
        
        # JointsMSELoss of supervised sample
        # Loss Pose
        loss_pose = self.mse_criterion(sup_ht, sup_target, sup_target_weight)

        # preds, maxvals = get_max_preds_tensor(unsup_ht.detach())
     
        # JointsMSELoss of unsupervised sample
        # if not self.cfg.MULTI_AUGS:
        if (not self.cfg.MULTI_AUGS) and (not self.cfg.REPEAT_AUG):
            loss_cons = self.mse_criterion(cons_ht, unsup_ht.detach(), unsup_target_weight, unsup_flag=True)
        else:
            loss_cons = 0
            if self.cfg.HEAT_FUSION:
                cons_ht_avg = torch.stack(cons_ht_list, dim=0).sum(0) / len(cons_ht_list)
                loss_cons += self.mse_criterion(cons_ht_avg, unsup_ht.detach(), unsup_target_weight, unsup_flag=True)
            else:
                for cons_ht in cons_ht_list:
                    loss_cons += self.mse_criterion(cons_ht, unsup_ht.detach(), unsup_target_weight, unsup_flag=True)

        loss = loss_pose + loss_cons

        loss_dic = {
            'loss_pose': loss_pose,
            'loss_cons': loss_cons,
        }

        return loss, loss_dic

