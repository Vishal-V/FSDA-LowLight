from __future__ import division

import os, scipy.io
import numpy as np
import logging
import argparse
import sys
import time
import cv2
import gc

# Import the Dataset class
from data import SonyDataset, NikonTrainSet

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch import Tensor
from model import Unet
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from datetime import datetime
from model import Task_filter

# Include parallel gradient flow to respective workers?
from multiprocessing import Manager

from itertools import cycle
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
os.environ["OMP_NUM_THREADS"]="8" # Maybe use "1"?
logging.getLogger('matplotlib.font_manager').disabled = True

manual_seed = 1
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.backends.cudnn.benchmark=False


def cos_loss(img1, img2):
    img1 = F.normalize(img1.type('torch.cuda.FloatTensor'), p=2, dim=1)
    img2 = F.normalize(img2.type('torch.cuda.FloatTensor'), p=2, dim=1)
    return torch.mean(1.0 - F.cosine_similarity(img1, img2))

def ssim_grayscale(img1, img2, ssim_fun, device):
    img1 = np.transpose(img1.data.cpu().numpy()[0], (1,2,0))
    img2 = np.transpose(img2.data.cpu().numpy()[0], (1,2,0))

    img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2GRAY)
    img1 = torch.reshape(torch.from_numpy(img1), (1,1,img1.shape[0],img1.shape[1]))
    img1 = img1.type('torch.DoubleTensor')

    img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_BGR2GRAY)
    img2 = torch.reshape(torch.from_numpy(img2), (1,1,img2.shape[0],img2.shape[1]))
    img2 = img2.type('torch.DoubleTensor')

    return ssim_fun(img1.to(device), img2.to(device))

def rgb2gray(img1, img2, device):
    img1 =  0.299 * img1[0][0].unsqueeze(dim=0).unsqueeze(dim=0)  + 0.587 * img1[0][1].unsqueeze(dim=0).unsqueeze(dim=0) + 0.114 * img1[0][2].unsqueeze(dim=0).unsqueeze(dim=0)
    img2 =  0.299 * img2[0][0].unsqueeze(dim=0).unsqueeze(dim=0)  + 0.587 * img2[0][1].unsqueeze(dim=0).unsqueeze(dim=0) + 0.114 * img2[0][2].unsqueeze(dim=0).unsqueeze(dim=0)
    img1 = img1.type('torch.FloatTensor')
    img2 = img2.type('torch.FloatTensor')

    return img1, img2    

def worker_init_fn(worker_id):                                                          
    np.random.seed()


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate(
        (
            im[0:H:2, 0:W:2, :],
            im[0:H:2, 1:W:2, :],
            im[1:H:2, 1:W:2, :],
            im[1:H:2, 0:W:2, :],
        ),
        axis=2,
    )

    # print (np.max(out))
    return out


def pack_nikon(raw, resize=False):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 600, 0) / (16383 - 600)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate(
        (
            im[0:H:2, 0:W:2, :],
            im[0:H:2, 1:W:2, :],
            im[1:H:2, 1:W:2, :],
            im[1:H:2, 0:W:2, :],
        ),
        axis=2,
    )
    if resize:
        out = cv2.resize(out, (out.shape[1] // 4, out.shape[0] // 4))

    return out


def pack_canon(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 2048, 0) / (16383 - 2048)  # Subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

def psnr(pred_image, gt_image):
    mse = np.mean(np.square(pred_image - gt_image))
    return 10 * np.log10(1 / mse.item())

def train(args, no_of_items, load_model=False):
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    print(device)
    
    if load_model:
        # Load models for target experiment 1
        # Load the model for set 1
        ckpt_1 = args.model_load_dir2 + f'model_{args.model_to_load2}_{args.num_target_samples_1}_1.pl'
        task_1 = Task_filter()
        checkpoint_1 = torch.load(ckpt_1)
        task_1.load_state_dict(checkpoint_1['task_state_dict'])
        task_1.to(device)
        log_1.info('Loaded weights for set 1')

    else:
        task_1 = Task_filter()
        task_1.to(device)

    # Load the weights for Sony 16 to 8 bit converter (Use 2k epochs from GT only converter)
    ckpt_jpeg = args.model_load_dir1 + 'task_%s.pl' % args.model_to_load1
    jpeg = Unet()
    resume_jpeg = torch.load(ckpt_jpeg)
    jpeg.load_state_dict(resume_jpeg['state_dict'])
    jpeg.to(device)

    # Training data for experiment 1
    source_trainset = SonyDataset(args.source_input_dir, args.source_gt_dir, args.ps)
    target_trainset_1 = NikonTrainSet(args.target_input_dir, args.target_gt_dir)

    source_train_loader = DataLoader(source_trainset, batch_size=args.batch_size, shuffle=True, 
                        num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    args.log_interval = len(source_train_loader)

    target_trainloader_1 = DataLoader(target_trainset_1, batch_size=args.batch_size, shuffle=True, 
                        num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)

    log_1.info("Length of source train loader : %d" % len(source_train_loader))
    log_1.info(f'Length of target train loader for set 1 {args.num_target_samples_1} images: {len(target_trainloader_1)}')

    # Loss function
    criterion = nn.L1Loss()
    criterion = criterion.to(device)

    # Optimizers
    optim_task_1 = optim.Adam(task_1.parameters(), lr=args.task_lr, weight_decay=args.wd)
    
    # LR schedulers
    task_scheduler_set1 = optim.lr_scheduler.StepLR(optim_task_1, step_size=2000, gamma=0.1)

    log_1.info("Target train loaders initialized... ")
    log_1.info('4k epochs. Step LR decay every 2k epochs. 16-bit source cos loss + \
        8-bit source ssim loss. update. 16-bit l1 target loss. update.')

    # Training
    source_loss_list_set1 = []
    target_loss_list_set1 = []

    for epoch in range(args.start_epoch+1, args.num_epoch):
        
        st = time.time()

        # Assuming training 2 experiments
        log_1.info("_____________ epoch %d started _______________"%epoch)
        
        running_source_loss_set1 = 0.0
        running_source_ssim_set1 = 0.0
        running_source_cos_set1 = 0.0
        running_target_loss_set1 = 0.0

        for i, data in enumerate(zip(source_train_loader, 
                                cycle(target_trainloader_1))):

            source_batch, target_batch_set1 = data 
            source_input_patch, source_gt_patch, source_train_id, source_ratio  = source_batch

            if epoch == 0:
                log_1.info ('[%d] source train id : %d, ratio : %g' % ( i, source_train_id.data.numpy(), source_ratio.data.numpy()))
            source_input_patch, source_gt_patch = source_input_patch.to(device), source_gt_patch.to(device)

            target_input_patch_set1, target_gt_patch_set1, target_train_id_set1, target_ratio_set1  = target_batch_set1

            if epoch == 0:
                log_1.info ('[%d] Target train id set1: %d, ratio : %g' % (i, target_train_id_set1.data.numpy(), target_ratio_set1.data.numpy())) 

            source_input_patch, source_gt_patch = source_input_patch.type('torch.cuda.FloatTensor'), source_gt_patch.type('torch.cuda.FloatTensor')
            source_input_patch, source_gt_patch = source_input_patch.to(device), source_gt_patch.to(device)
            
            target_input_patch_set1, target_gt_patch_set1 = target_input_patch_set1.type('torch.cuda.FloatTensor'), target_gt_patch_set1.type('torch.cuda.FloatTensor')
            target_input_patch_set1, target_gt_patch_set1 = target_input_patch_set1.to(device), target_gt_patch_set1.to(device)

            # Experiment 1
            # Set 1 Source Training
            optim_task_1.zero_grad()
            source_outputs_1 = task_1(source_input_patch, True)
            source_cosine_loss_1 = cos_loss(source_outputs_1, source_gt_patch)

            source_gt_patch_jpeg = jpeg(source_gt_patch)
            source_outputs_jpeg_1 = jpeg(source_outputs_1)
            source_ssim_loss_1, _ = ssim(source_outputs_jpeg_1, source_gt_patch_jpeg, data_range = 1, size_average=True)
            source_ssim_loss_1 = 1.0 - source_ssim_loss_1
            
            source_loss_set1 = source_ssim_loss_1 + source_cosine_loss_1
            source_loss_set1.backward()
            optim_task_1.step()

            try:
                # Set 1 target Training
                optim_task_1.zero_grad()
                target_outputs_1 = task_1(target_input_patch_set1, False)
                target_loss_1 = criterion(target_outputs_1, target_gt_patch_set1)
                target_loss_1.backward()
                optim_task_1.step()
            except:
                print(f'Type mismatch...')
                continue

            # Print loss statistics
            running_source_loss_set1 += source_loss_set1.item()
            running_target_loss_set1 += target_loss_1.item()

            running_source_ssim_set1 += source_ssim_loss_1.item()
            running_source_cos_set1 += source_cosine_loss_1.item()

        print("GPU time : %s" % (time.time() - st))
        st1 = time.time()

        source_loss_list_set1.append(running_source_loss_set1 / args.log_interval)
        target_loss_list_set1.append(running_target_loss_set1 / args.log_interval)

        log_1.info(' [%d] Source training loss set1: %.4f %s' % (epoch, running_source_loss_set1 / args.log_interval, datetime.now()))
        log_1.info(' [%d] Target training loss set1: %.4f %s' % (epoch, running_target_loss_set1 / args.log_interval, datetime.now()))
        log_1.info('<------------------------------------------------------------------>')

        if epoch % 10 == 0:
            log_gen.info(' [%d] Source training loss set1: %.4f %s' % (epoch, running_source_loss_set1 / args.log_interval, datetime.now()))
            log_gen.info(' [%d] Target training loss set1: %.4f %s' % (epoch, running_target_loss_set1 / args.log_interval, datetime.now()))
            log_gen.info('<------------------------------------------------------------------>')
        
        div = args.log_interval
        loss_1.info(f'{running_source_ssim_set1/div}, {running_source_cos_set1/div}, {running_source_loss_set1/div}, {running_target_loss_set1/div}')

        task_scheduler_set1.step()

        # Save model at snapshot epoch
        if epoch % args.model_save_freq == 0:
            state = {'epoch': epoch , 'task_state_dict': task_1.state_dict(), 'task_optimizer': optim_task_1.state_dict(), 'task_scheduler':task_scheduler_set1.state_dict() }
            torch.save(state, args.checkpoint_dir + 'model_%d_%d_%d_sony_nikon.pl' % (epoch, args.num_target_samples_1, 1))

            log_1.info(f'Saved model at epoch: {epoch}')


def set_logger(args, logger_name, t_set, exp=0, task=0, track=None, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    if exp == 1:
        fileHandler = logging.FileHandler(os.path.join(args.result_dir, f'log_Sony_nikon_{args.num_target_samples_1}_{t_set}.txt'), mode='a')
    elif exp == 2:
        fileHandler = logging.FileHandler(os.path.join(args.result_dir, f'log_Sony_nikon_{args.num_target_samples_2}_{t_set}.txt'), mode='a')
    elif logger_name == 'general':
        fileHandler = logging.FileHandler(os.path.join(args.result_dir, f'log_Sony_nikon_general_{args.num_target_samples_1}.txt'), mode='a')
    elif exp == 3:
        fileHandler = logging.FileHandler(os.path.join(args.val_result_dir, f'loss_Sony_nikon_{args.num_target_samples_1}_{task}.txt'), mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="command for training p3d network")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--source_input_dir', type=str, default='./dataset/Sony/Sony/short/')
    parser.add_argument('--source_gt_dir', type=str, default='./dataset/Sony/Sony/long/')
    parser.add_argument('--target_input_dir', type=str, default='./dataset/Nikon/short/')
    parser.add_argument('--target_gt_dir', type=str, default='./dataset/Nikon/long_png/')
    
    parser.add_argument('--model_load_dir1', type=str, default='./checkpoint/Sony/bit_change/')
    parser.add_argument('--model_to_load1', type=str, default='2000_1')
    parser.add_argument('--model_load_dir2', type=str, default='./checkpoint/Sony_Nikon/approach3/training/')
    parser.add_argument('--model_to_load2', type=str, default='500')
    
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/Sony_Nikon/approach3/')
    parser.add_argument('--result_dir', type=str, default='./result_Sony/Sony_Nikon/approach3/')

    parser.add_argument('--val_result_dir', type=str, default='./result_Sony/Sony_Nikon/approach3/validation/')
    parser.add_argument('--validation_input_dir', type=str, default='./dataset/Nikon/short_new/')
    parser.add_argument('--validation_gt_dir', type=str, default='./dataset/Nikon/long_new/')

    parser.add_argument('--num_workers', type=int, default=0, help='multi-threads for data loading')
    parser.add_argument('--ps', type=int, default=512)
    parser.add_argument('--log_interval', type=int, default=161)
    parser.add_argument('--psnr_log_interval', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--error_plot_freq', type=int, default=200)
    parser.add_argument('--task_lr', type=float, default=1e-4)
    parser.add_argument('--disc_lr', type=float, default=5*1e-4)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=4001)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--model_save_freq', type=int, default=100)
    parser.add_argument('--num_target_samples_1', type=int, default=1)
    parser.add_argument('--num_experiments', type=int, default=1)
    parser.add_argument('--validation_epoch', type=int, default=250)

    args = parser.parse_args()

    # Create output dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    if not os.path.exists(args.val_result_dir):
        os.makedirs(args.val_result_dir)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # Set Logger
    set_logger(args, 'log_set1', t_set=1, exp=1)
    set_logger(args, 'general', t_set=1)
    set_logger(args, 'loss_1', exp=3, task=1, t_set=1)
    
    log_1 = logging.getLogger('log_set1')
    log_gen = logging.getLogger('general')
    loss_1 = logging.getLogger('loss_1')

    # Start training
    log_1.info(" ".join(sys.argv))
    log_1.info(args)
    log_1.info("Using device %s" % str(args.gpu))

    for count in range(1, 2):
        train(args, count, load_model=False)
    print('Completed training')
