# ------------------------------------------------------------------------------
# Written by Huayi Zhou (sjtu_zhy@sjtu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import torch
import numpy as np

import torchvision.transforms as transforms


########################################################################

# YOCO(ICML2022) You Only Cut Once: Boosting Data Augmentation with a Single Cut
# https://github.com/JunlinHan/YOCO
def random_YOCO(images, images_aux):  # images or images_aux is processed by RandAug
    N,_,w,h = images.shape
    
    for i in range(N):
        if torch.rand(1) > 0.5:
            # images[i, :, 0:int(w/2), :] = images_aux[i, :, 0:int(w/2), :]  # also right
            images[i, :, int(w/2):w, :] = images_aux[i, :, int(w/2):w, :]
        else:
            # images[i, :, :, 0:int(h/2)] = images_aux[i, :, :, 0:int(h/2)]  # also right
            images[i, :, :, int(h/2):h] = images_aux[i, :, :, int(h/2):h]
    return images

########################################################################
    
# arxiv2017.08 (Cutout) Improved Regularization of Convolutional Neural Networks with Cutout
# https://github.com/uoguelph-mlrg/Cutout
def random_cutout(image, MASK_HOLES_NUM=4):
    N,_,width,height = image.shape
    center_x = torch.randint(0,width,(N,MASK_HOLES_NUM)).int().cuda()
    center_y = torch.randint(0,height,(N,MASK_HOLES_NUM)).int().cuda()
    size = torch.randint(10,20,(N,MASK_HOLES_NUM,2)).int().cuda()
    
    x0 = torch.clamp_(center_x-size[...,0],0,width)
    y0 = torch.clamp_(center_y-size[...,1],0,height)

    x1 = torch.clamp_(center_x+size[...,0],0,width)
    y1 = torch.clamp_(center_y+size[...,1],0,height)

    for i in range(N):
        for j in range(MASK_HOLES_NUM):
            image[i, :, y0[i,j]:y1[i,j], x0[i,j]:x1[i,j]] = 0
    return image

    
# ICLR2018 Mixup - Beyond Empirical Risk Minimization
# https://github.com/facebookresearch/mixup-cifar10
def random_mixup(image):
    alpha = 0.2
    lam = np.random.beta(alpha, alpha)  # lam in (0 ,1)
    
    N,_,width,height = image.shape
    rand_index = torch.randperm(N).cuda()
    image_rand = image[rand_index]
    
    lam = 1 - lam * 0.25  # we want the lam in (0.75, 1), and keep most features of the original image
    
    # for i in range(N):
        # mixed_image = lam * image[i, ...] + (1 - lam) * image_rand[i, ...]
        # image[i, ...] = mixed_image
    # return image
    
    mixed_image = lam * image + (1 - lam) * image_rand
    return mixed_image


# ICCV2019 CutMix - Regularization Strategy to Train Strong Classifiers With Localizable Features
# https://github.com/clovaai/CutMix-PyTorch
def random_cutmix(image, MASK_HOLES_NUM=2):
    N,_,width,height = image.shape
    center_x = torch.randint(0,width,(N,MASK_HOLES_NUM)).int().cuda()
    center_y = torch.randint(0,height,(N,MASK_HOLES_NUM)).int().cuda()
    size = torch.randint(10,20,(N,MASK_HOLES_NUM,2)).int().cuda()
    
    x0 = torch.clamp_(center_x-size[...,0],0,width)
    y0 = torch.clamp_(center_y-size[...,1],0,height)

    x1 = torch.clamp_(center_x+size[...,0],0,width)
    y1 = torch.clamp_(center_y+size[...,1],0,height)
    
    rand_index = torch.randperm(N).cuda()
    image_rand = image[rand_index]

    for i in range(N):
        for j in range(MASK_HOLES_NUM):
            image[i, :, y0[i,j]:y1[i,j], x0[i,j]:x1[i,j]] = image_rand[i, :, y0[i,j]:y1[i,j], x0[i,j]:x1[i,j]]
    return image


# ICCV2021 An Empirical Study of the Collapsing Problem in Semi-Supervised 2D Human Pose Estimation
# https://github.com/xierc/Semi_Human_Pose (Joint Cutout)
def mask_joint(image,joints,MASK_JOINT_NUM=4):
    ## N,J,2 joints
    N,J = joints.shape[:2]
    _,_,width,height = image.shape
    re_joints = joints[:,:,:2] + torch.randn((N,J,2)).cuda()*10
    re_joints = re_joints.int()
    size = torch.randint(10,20,(N,J,2)).int().cuda()

    x0 = re_joints[:,:,0]-size[:,:,0]
    y0 = re_joints[:,:,1]-size[:,:,1]

    x1 = re_joints[:,:,0]+size[:,:,0]
    y1 = re_joints[:,:,1]+size[:,:,1]

    torch.clamp_(x0,0,width)
    torch.clamp_(x1,0,width)
    torch.clamp_(y0,0,height)
    torch.clamp_(y1,0,height)

    for i in range(N):
        # num = np.random.randint(MASK_JOINT_NUM)
        # ind = np.random.choice(J, num)
        ind = np.random.choice(J, MASK_JOINT_NUM)
        for j in ind:
            image[i,:,y0[i,j]:y1[i,j],x0[i,j]:x1[i,j]] = 0
    return image


# CVPR2023, Semi-Supervised 2D Human Pose Estimation Driven by Position Inconsistency Pseudo Label Correction Module
# https://github.com/hlz0606/SSPCM/blob/main/SSPCM/lib/models/my_pose_triple.py#L836
# Semi-supervised Cut-Occlude based on pseudo keypoint perception (SSCO). See Figure 5 in the original paper.
def keypoint_cutmix_perception(image, joints, MASK_JOINT_NUM=2):

    image_tmp = image.clone()

    ## N,J,2 joints
    N, J = joints.shape[:2]
    _, _, height, width = image.shape
    re_joints = joints[:, :, :2] + torch.randn((N, J, 2)).cuda() * 10
    re_joints = 0 + re_joints.int()
    size = torch.randint(10, 20, (N, J, 2)).int().cuda()

    center_x = copy.deepcopy(re_joints[:, :, 1])
    center_y = copy.deepcopy(re_joints[:, :, 0])

    x0 = re_joints[:, :, 1] - size[:, :, 1]
    y0 = re_joints[:, :, 0] - size[:, :, 0]

    x1 = re_joints[:, :, 1] + size[:, :, 1]
    y1 = re_joints[:, :, 0] + size[:, :, 0]

    x0 = torch.clamp(x0, 0, width-1)
    x1 = torch.clamp(x1, 0, width-1)
    y0 = torch.clamp(y0, 0, height-1)
    y1 = torch.clamp(y1, 0, height-1)

    for i in range(N):
        ind = np.random.choice(J, MASK_JOINT_NUM)
        ind_2 = np.random.choice(J, MASK_JOINT_NUM)    ######
        img_id = np.random.randint(0, N)    #####
        ##

        for idx in range(len(ind)):
            j = ind[idx]
            j2 = ind_2[idx]

            x_start = center_x[i, j] - abs(x0[img_id, j2] - center_x[img_id, j2])    #######
            x_end = center_x[i, j] + abs(x1[img_id, j2] - center_x[img_id, j2])
            y_start = center_y[i, j] - abs(y0[img_id, j2] - center_y[img_id, j2])  ######
            y_end = center_y[i, j] + abs(y1[img_id, j2] - center_y[img_id, j2])
   
            x_start = torch.clamp(x_start, 0, width-1)
            x_end = torch.clamp(x_end, 0, width-1)
            y_start = torch.clamp(y_start, 0, height-1)
            y_end = torch.clamp(y_end, 0, height-1)
            
            sub_img2 = image[img_id, :, y0[img_id, j2]:y1[img_id, j2], x0[img_id, j2]:x1[img_id, j2]]
            sub_img1 = image[i, :, y_start:y_end, x_start:x_end]
            offset_y = abs(sub_img2.shape[-2] - sub_img1.shape[-2])
            offset_x = abs(sub_img2.shape[-1] - sub_img1.shape[-1])
            
            offset_y_start = 0
            offset_x_start = 0
            offset_y_end = 0
            offset_x_end = 0
            if y_start == 0:
                offset_y_start = offset_y
            if x_start == 0:
                offset_x_start = offset_x
            if y_end == height-1:
                offset_y_end = offset_y
            if x_end == width-1:
                offset_x_end = offset_x


            if image[i, :, y_start:y_end, x_start:x_end].shape[-1] == 0 or \
                    image[i, :, y_start: y_end, x_start: x_end].shape[-2] == 0:  ##
                pass

            elif image[img_id, :, y0[img_id, j2] + offset_y_start: y1[img_id, j2] - offset_y_end,
                        x0[img_id, j2] + offset_x_start: x1[img_id, j2] - offset_x_end].shape[-1] == 0 or \
                    image[img_id, :, y0[img_id, j2] + offset_y_start: y1[img_id, j2] - offset_y_end,
                    x0[img_id, j2] + offset_x_start: x1[img_id, j2] - offset_x_end].shape[-2] == 0:  ##
                image[i, :, y_start:y_end, x_start:x_end] = 0

            elif image[i, :, y_start: y_end, x_start: x_end].shape != \
                    image[img_id, :, y0[img_id, j2] + offset_y_start: y1[img_id, j2] - offset_y_end,
                        x0[img_id, j2] + offset_x_start: x1[img_id, j2] - offset_x_end].shape:
                image[i, :, y_start:y_end, x_start:x_end] = 0

            else:    ### keypoint cutmix
                image[i, :, y_start:y_end, x_start: x_end] = \
                        image_tmp[img_id, :, y0[img_id, j2] + offset_y_start: y1[img_id, j2] - offset_y_end,
                            x0[img_id, j2] + offset_x_start: x1[img_id, j2] - offset_x_end]

    return image
    
