# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2

from fcn import VGGNet, FCN32s, FCN8s

#from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os
sys.path.append("../../")
from PennFudanDataset_main import *

#import pdb
#pdb.set_trace()

def collate_fn(batch):
    return tuple(zip(*batch))

n_class    = 2

batch_size = 6
epochs     = 500
lr         = 1e-4
momentum   = 0
w_decay    = 1e-5
step_size  = 50
gamma      = 0.5
configs    = "FCNs-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)


# create dir for model
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)

use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))

# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations
dataset = PennFudanDataset('data_sets/PennFudanPed', get_transform(train=True))
dataset_test = PennFudanDataset('data_sets/pennFudanPed', get_transform(train=False))

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=collate_fn)

val_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=3, shuffle=False, num_workers=4,
        collate_fn=collate_fn)

print("That's it!")


#vgg_model = VGGNet(requires_grad=True, remove_fc=True)
print("not use pretrained VGG model")
vgg_model = VGGNet(pretrained=False,requires_grad=True, remove_fc=True)
fcn_model = FCN8s(pretrained_net=vgg_model, n_class=n_class)

if use_gpu:
    ts = time.time()
    vgg_model = vgg_model.cuda()
    fcn_model = fcn_model.cuda()
    fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs
#optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)

# create dir for score
score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
IU_scores    = np.zeros((epochs, n_class))
pixel_scores = np.zeros(epochs)


def train():
    for epoch in range(epochs):
        ts = time.time()
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = input_process(batch)
            _,labels = target_process(batch)


            #if use_gpu:
            #    inputs = Variable(batch['X'].cuda())
            #    labels = Variable(batch['Y'].cuda())
            #else:
            #    #inputs, labels = Variable(batch['X']), Variable(batch['Y'])
            #    inputs, labels = Variable(batch['X']), Variable(batch['Y'])

            outputs = fcn_model(inputs)
            #import pdb
            #pdb.set_trace()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.data.item()))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(fcn_model, model_path)

        val(epoch)
        scheduler.step()



def input_process(batch):
    #import pdb
    #pdb.set_trace()
    batch_size=len(batch[0])
    input_batch= torch.zeros(batch_size,3,160,160)
    for i in range(batch_size): 
        inputs_tmp = Variable(batch[0][i])
        inputs_tmp1=cv2.resize(inputs_tmp.permute([1,2,0]).numpy(),(160,160))
        inputs_tmp2=torch.tensor(inputs_tmp1).permute([2,0,1])
        input_batch[i:i+1,:,:,:]= torch.unsqueeze(inputs_tmp2,0)
    return input_batch 


def target_process(batch):
    batch_size=len(batch[0])
    target_batch= torch.zeros(batch_size,160,160)
    label_batch = torch.zeros(batch_size,2,160,160)
    for i in range(batch_size):
        #只处理batch中的第一张图片
        masks=batch[1][i]['masks']
        masks1=torch.sum(masks,0)
        masks2=(masks1>0).float()
        masks3=torch.tensor(cv2.resize(masks2.numpy(),(160,160)))
        target_batch[i:i+1,:,:] = torch.unsqueeze(masks3,0)
        label_batch[i:i+1,1,:,:] =torch.unsqueeze(masks3,0)
        label_batch[i:i+1,0,:,:] =torch.unsqueeze(1-masks3,0)
        #target = batch['l'].cpu().numpy().reshape(N, h, w)
    return target_batch.numpy(),label_batch

def val(epoch):
    fcn_model.eval()
    total_ious = []
    pixel_accs = []
    for iter, batch in enumerate(val_loader):
        inputs = input_process(batch)
        target,label= target_process(batch)

        output = fcn_model(inputs)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
        
        for p, t in zip(pred, target):
            total_ious.append(iou(p, t))
            pixel_accs.append(pixel_acc(p, t))

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
    IU_scores[epoch] = ious
    np.save(os.path.join(score_dir, "meanIU"), IU_scores)
    pixel_scores[epoch] = pixel_accs
    np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)


# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions
def iou(pred, target):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    #import pdb
    #pdb.set_trace()
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total


if __name__ == "__main__":
    val(0)  # show the accuracy before training
    train()
