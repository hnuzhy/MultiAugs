# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Huayi Zhou (sjtu_zhy@sjtu.edu.cn)
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import argparse
import numpy as np
import cv2
import random

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import dataset
import models

from utils.augments import random_cutout
from utils.augments import random_mixup
from utils.augments import random_cutmix
from utils.augments import mask_joint
from utils.augments import keypoint_cutmix_perception

from utils.transforms import get_affine_transform

from core.config import config
from core.config import update_config
from core.config import update_dir
from core.inference import get_max_preds_tensor


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
                        
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)

    parser.add_argument('--num_gpus',
                        help='num_gpus for DDP',
                        type=str)

    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)
                        
    parser.add_argument('--model-file',
                        help='model state file',
                        type=str)
                        
    parser.add_argument('--local_rank',
                        help='num of dataloader workers',
                        type=int)

    parser.add_argument(
        '--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument(
        '--dataDir', help='data directory', type=str, default='')
    parser.add_argument(
        '--data-format', help='data format', type=str, default='')

    parser.add_argument('--NoDebug', type=str, default='', 
                       help='create model without Debug')
    parser.add_argument('--pretrained_model', type=str, default='', 
                       help='The path of pretrained model')

    parser.add_argument(
        '--distributed',
        action='store_true',
        help='whether using distributed training')


    parser.add_argument('--batch_size',
                        help='num of batch size of each step for one GPU',
                        type=int)
    parser.add_argument('--epochs',
                        help='num of total epochs for training',
                        type=int)
    parser.add_argument('--lr_steps', type=int, nargs='+', default=[70, 90])
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    

    parser.add_argument('--ra_aug', action='store_true', help='whether using random augment (RA)')
    parser.add_argument('--jc_aug', action='store_true', help='whether using Joint Cutout (JC)')
    parser.add_argument('--jo_aug', action='store_true', help='whether using Joint CutMix (Cut-Occlude, JO)')
    parser.add_argument('--co_aug', action='store_true', help='whether using Cutout')
    parser.add_argument('--cm_aug', action='store_true', help='whether using CutMix')
    parser.add_argument('--mu_aug', action='store_true', help='whether using Mixup')
    parser.add_argument('--jccm_aug', action='store_true', help='whether using JointCutout and CutMix')
    parser.add_argument('--joco_aug', action='store_true', help='whether using JointCutMix and Cutout')


    parser.add_argument(
        '--exp_subname', help='exp results directory with subname', type=str, default='')
    parser.add_argument(  # implemented; can not improve and be harmful
        '--prog_mode', help='the mode of progressively updating PROG_X', type=str, default='')  # do not work :-(
    parser.add_argument(  # implemented; but can not improve mAP either
        '--stepwise', action='store_true', help='progressively updating by stepwise, not epochwise')  # do not work :-(
    parser.add_argument(  # implemented; have been verfied to be work by many ablation studies!
        '--multi_augs', action='store_true', help='unsup-losses by applying multi-augs separately')  # can work !!!
    parser.add_argument(  # implemented; can not work for leading to zero losses!!!
        '--sharpen_pred', action='store_true', help='regularizing predictions to have low entropy')  # do not work :-(
    parser.add_argument(  # implemented; can not work for giving worse results than our original separare design.
        '--heat_fusion', action='store_true', help='sum + avg fusion of multiple predicted heatmaps')  # do not work :-(
    parser.add_argument(  # implemented; can not work for improving due to filtering out low-conf but vital results.
        '--conf_masking', action='store_true', help='mask out examples that being not confident')  # do not work :-(
    parser.add_argument(
        '--conf_thresh', type=float, default=0.5, help='confident threshold for conf_masking')

    parser.add_argument(
        '--train_len', help='training length of labeled images', type=int)
        
    args = parser.parse_args()
    update_dir(args.modelDir, args.logDir, args.dataDir)

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.data_format:
        config.DATASET.DATA_FORMAT = args.data_format
    if args.NoDebug:
        config.DEBUG.DEBUG = False
        config.TEST.SAVE_RESULT = False
    if args.pretrained_model:
        config.MODEL.PRETRAINED = args.pretrained_model
        
    if args.batch_size:
        config.TRAIN.BATCH_SIZE = args.batch_size
    if args.epochs:
        config.TRAIN.END_EPOCH = args.epochs
    if args.lr_steps:
        config.TRAIN.LR_STEP = args.lr_steps
    if args.lr:
        config.TRAIN.LR = args.lr
        
    # config.USE_RandAug_AUG = args.ra_aug
    config.CONS_RAND_AUG = args.ra_aug
    config.USE_JointCutout_AUG = args.jc_aug
    config.USE_JointCutMix_AUG = args.jo_aug
    config.USE_Cutout_AUG = args.co_aug
    config.USE_CutMix_AUG = args.cm_aug
    config.USE_MixUp_AUG = args.mu_aug
   
    config.USE_JointCutMix_CutOut_AUG = args.joco_aug  # used when args.multi_augs is True
    config.USE_JointCutout_CutMix_AUG = args.jccm_aug  # used when args.multi_augs is True
    
    config.EXP_SUB_NAME = args.exp_subname
    config.PROG_MODE = args.prog_mode
    config.STEP_WISE = args.stepwise
    config.MULTI_AUGS = args.multi_augs
    config.SHARPEN_PRED = args.sharpen_pred
    config.HEAT_FUSION = args.heat_fusion
    config.CONF_MASKING = args.conf_masking
    config.CONF_THRESH = args.conf_thresh
    
    if args.train_len:
        config.DATASET.TRAIN_LEN = args.train_len


    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file


def get_batch_affine_transform(batch_size):

    sf = config.DATASET.SCALE_FACTOR
    rf = config.DATASET.ROT_FACTOR
    batch_trans  = []
    for b in range(batch_size):
        r = 0
        s = 1
        c = config.MODEL.IMAGE_SIZE/2
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
        r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.8 else 0

        trans = cv2.getRotationMatrix2D((0,0), r, s)
        batch_trans.append(trans)

    batch_trans = np.stack(batch_trans, 0)
    batch_trans = torch.from_numpy(batch_trans).cuda()

    return batch_trans



def main():
    
    '''step 1'''
    args = parse_args()
    reset_config(config, args)
    
    gpus = [int(i) for i in config.GPUS.split(',')]
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = eval('dataset.'+config.DATASET.TRAIN_DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TRAIN_SET,
        True,
        train_transforms,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
    )


    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=False
    )
    gpus = [int(i) for i in config.GPUS.split(',')]
    if config.TEST.MODEL_FILE:
        model_state = torch.load(config.TEST.MODEL_FILE, map_location='cpu')
        if isinstance(model_state, dict) and 'state_dict' in model_state:
            print('checkpoint')
            model_state = model_state['state_dict']
            
        if list(model_state.keys())[0][:6] == 'module':
            model_state = {k[7:]:v for k,v in model_state.items()}
        
        if list(model_state.keys())[0][:6] == 'resnet':
            if config.MODEL.NAME in ['pose_resnet','pose_hrnet']:
                model_state = {k[7:]:v for k,v in model_state.items()}
            model.load_state_dict(model_state, strict=False)
        else:
            if config.MODEL.NAME in ['pose_resnet','pose_hrnet']:
                model.load_state_dict(model_state, strict=False)
            elif config.MODEL.NAME in ['pose_blend']:
                model.resnet.load_state_dict(model_state, strict=False)
    else: 
        model_state_file = os.path.join(final_output_dir, 'final_state.pth.tar')
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    
    '''
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    }
    '''
    kp_ids = [0, 7, 11]
    
    
    '''step 2'''
    for index, (input, target, target_weight, meta) in enumerate(train_loader):
        
        meta_sup, meta_unsup = meta[0], meta[1]
        
        output = model(input)  # input should be a list!!!
        
        # [sup_ht, unsup_ht, unsup_ht_trans, cons_x_final, cons_ht, theta] = output  # pose_cons
        [sup_ht, unsup_ht, unsup_ht_trans, cons_x_list_final, cons_ht_list, theta] = output  # pose_cons + MultiAugs
 
        # A30
        sup_x, unsup_x = input
        cons_x_A30 = unsup_x.clone().cuda()  # shape is (N,C,height,width)
        
        # get the A60 image
        grid = F.affine_grid(theta, cons_x_A30.size()).float()
        cons_x_A60 = F.grid_sample(cons_x_A30, grid)
            
        # the order should be the same as in pose_cons.py
        assert config.USE_JointCutout_AUG, "--jc_aug should be added!"
        cons_x_JC, cons_ht_JC = cons_x_list_final[0], cons_ht_list[0]
        
        assert config.USE_JointCutMix_AUG, "--jo_aug should be added!"
        cons_x_JO, cons_ht_JO = cons_x_list_final[1], cons_ht_list[1]
        
        assert config.USE_JointCutMix_CutOut_AUG, "--joco_aug should be added!"
        cons_x_JOCO, cons_ht_JOCO = cons_x_list_final[2], cons_ht_list[2]
        
        assert config.USE_JointCutout_CutMix_AUG, "--jccm_aug should be added!"
        cons_x_JCCM, cons_ht_JCCM = cons_x_list_final[3], cons_ht_list[3]


        # save cons_x_A60, cons_x_JC, cons_x_JO, cons_x_JOCO and cons_x_JCCM into images
        cons_x_A60 = cons_x_A60.cpu().numpy().transpose((0, 2, 3, 1))
        cons_x_A60 = np.array(np.clip(255 * (cons_x_A60 * std + mean), 0, 255), dtype=np.uint8)

        
        cons_x_JC = cons_x_JC.cpu().numpy().transpose((0, 2, 3, 1))
        cons_x_JC = np.array(np.clip(255 * (cons_x_JC * std + mean), 0, 255), dtype=np.uint8)
        cons_x_JO = cons_x_JO.cpu().numpy().transpose((0, 2, 3, 1))
        cons_x_JO = np.array(np.clip(255 * (cons_x_JO * std + mean), 0, 255), dtype=np.uint8)
        cons_x_JOCO = cons_x_JOCO.cpu().numpy().transpose((0, 2, 3, 1))
        cons_x_JOCO = np.array(np.clip(255 * (cons_x_JOCO * std + mean), 0, 255), dtype=np.uint8)
        cons_x_JCCM = cons_x_JCCM.cpu().numpy().transpose((0, 2, 3, 1))
        cons_x_JCCM = np.array(np.clip(255 * (cons_x_JCCM * std + mean), 0, 255), dtype=np.uint8)
        
        heatmaps_list = [unsup_ht_trans.cpu().numpy(),
            cons_ht_JC.detach().cpu().numpy(), cons_ht_JO.detach().cpu().numpy(),
            cons_ht_JOCO.detach().cpu().numpy(), cons_ht_JCCM.detach().cpu().numpy()]
        
        for img_idx in range(config.TRAIN.BATCH_SIZE):
        
            img_path = meta_unsup['image'][img_idx]
            center = meta_unsup['center'][img_idx].cpu().numpy()
            scale = meta_unsup['scale'][img_idx].cpu().numpy()
            print(index, "\t", img_idx, "\t", img_path, center, scale)
            data_numpy = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            
            c, s, r = center, scale, 0
            trans = get_affine_transform(c, s, r, config.MODEL.IMAGE_SIZE)
            input_ori = cv2.warpAffine(
                data_numpy,
                trans,
                (int(config.MODEL.IMAGE_SIZE[0]), int(config.MODEL.IMAGE_SIZE[1])),
                flags=cv2.INTER_LINEAR)
            cv2.imwrite("./tmp/coco_train2017_MA_%d_%d_A0.jpg"%(index, img_idx), input_ori)
            
            cv2.imwrite("./tmp/coco_train2017_MA_%d_%d_A60.jpg"%(index, img_idx), cons_x_A60[img_idx, ...])
            cv2.imwrite("./tmp/coco_train2017_MA_%d_%d_JC.jpg"%(index, img_idx), cons_x_JC[img_idx, ...])
            cv2.imwrite("./tmp/coco_train2017_MA_%d_%d_JO.jpg"%(index, img_idx), cons_x_JO[img_idx, ...])
            cv2.imwrite("./tmp/coco_train2017_MA_%d_%d_JOCO.jpg"%(index, img_idx), cons_x_JOCO[img_idx, ...])
            cv2.imwrite("./tmp/coco_train2017_MA_%d_%d_JCCM.jpg"%(index, img_idx), cons_x_JCCM[img_idx, ...])
            
            # heatmap shape is [batch_size, num_joints, height, width]
            '''v1'''
            # for kpi in kp_ids: 
                # hm_type_list = ["A60", "JC", "JO", "JOCO", "JCCM"]
                # for heatmaps, hm_type in zip(heatmaps_list, hm_type_list):
                    # heatmap_kpi = heatmaps[img_idx, kpi, ...]
                    # max_len = heatmap_kpi.max() - heatmap_kpi.min()
                    # heatmap_kpi = (heatmap_kpi - heatmap_kpi.min()) / max_len
                    # heatmap_kpi = np.array(heatmap_kpi * 255, dtype=np.uint8)
                    # cv2.imwrite("./tmp/coco_train2017_MA_%d_%d_%s_kp%d.jpg"%(index, img_idx, hm_type, kpi), heatmap_kpi)
            '''v2'''
            hm_type_list = ["A60", "JC", "JO", "JOCO", "JCCM"]
            easy_heatmap = None
            for hid, (heatmaps, hm_type) in enumerate(zip(heatmaps_list, hm_type_list)):
                heatmap_kps = heatmaps[img_idx, :, ...]
                heatmap_avg = heatmap_kps.sum(0) / heatmaps.shape[1]  # channels average
                max_len = heatmap_avg.max() - heatmap_avg.min()
                heatmap_avg = (heatmap_avg - heatmap_avg.min()) / max_len  # normalize
                heatmap_avg = np.array(heatmap_avg * 255, dtype=np.uint8) # expand
                cv2.imwrite("./tmp/coco_train2017_MA_%d_%d_%s_kps.jpg"%(index, img_idx, hm_type), heatmap_avg)
                if hid == 0:
                    easy_heatmap = heatmap_avg
                else:
                    heatmap_diff = np.abs(heatmap_avg - easy_heatmap)
                    cv2.imwrite("./tmp/coco_train2017_MA_%d_%d_%s_kpsDiff.jpg"%(index, img_idx, hm_type), heatmap_diff)
                
        if index == 1:  break
                  
        
if __name__ == '__main__':
    main()


'''
using pose_cons_18/256x192_COCO1K_PoseCons_AS_baseline3_COCO10K_e100
$ python pose_estimation/plot_MultiHeats.py \
    --cfg experiments/mix_coco_coco/res18/256x192_COCO1K_PoseCons_AS.yaml \
    --gpus 0 --batch_size 32 --workers 8 --multi_augs --joco_aug --jccm_aug --jc_aug --jo_aug \
    --model-file output/mix_coco_coco/pose_cons_18/256x192_COCO1K_PoseCons_AS_baseline3_COCO10K_e100/model_best.pth.tar
'''

