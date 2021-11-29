#coding:utf-8
from PennFudanDataset_main import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from v1yolomodel import *
import cv2
import numpy as np
import time
import sys
import os


## 数据处理
#服务器上的地址 /data/2020-722-YOLOV4-Practical-datasets/PenFudanPed
# dataset地址：/Users/zhaomignming/Documents/mmteacher/datasets
#datapath='/Users/zhaomignming/Documents/mmteacher/datasets/PennFudanPed'
datapath='/Users/zhaomingming/data_sets/PennFudanPed'
dataset = PennFudanDataset(datapath, get_transform(train=False))
dataset_test = PennFudanDataset(datapath, get_transform(train=False))

indices = torch.randperm(len(dataset)).tolist()
#dataset = torch.utils.data.Subset(dataset, indices[:-50])
#dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
#dataset = torch.utils.data.Subset(dataset, indices[0:1])
#import pdb
#pdb.set_trace()
#dataset = torch.utils.data.Subset(dataset, indices[0:1])
dataset = torch.utils.data.Subset(dataset, [0])
dataset_test = torch.utils.data.Subset(dataset_test, indices[0:2])

def collate_fn(batch):
    return tuple(zip(*batch))
# define training and validation data loaders
train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=collate_fn)

val_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=False, num_workers=4,
        collate_fn=collate_fn)


def input_process(batch):
    #import pdb
    #pdb.set_trace()
    batch_size=len(batch[0])
    input_batch= torch.zeros(batch_size,3,448,448)
    for i in range(batch_size):
        inputs_tmp = Variable(batch[0][i])
        inputs_tmp1=cv2.resize(inputs_tmp.permute([1,2,0]).numpy(),(448,448))
        inputs_tmp2=torch.tensor(inputs_tmp1).permute([2,0,1])
        input_batch[i:i+1,:,:,:]= torch.unsqueeze(inputs_tmp2,0)
    return input_batch 

#batch[1][0]['boxes'][0]
def target_process(batch,grid_number=7):
    # batch[1]表示label
    # batch[0]表示image
    batch_size=len(batch[0])
    target_batch= torch.zeros(batch_size,grid_number,grid_number,5)
    #import pdb
    #pdb.set_trace()
    for i in range(batch_size):
        labels = batch[1]
        batch_labels = labels[i]
        #import pdb
        #pdb.set_trace()
        number_box = len(batch_labels['boxes'])
        for wi in range(grid_number):
            for hi in range(grid_number):
                # 遍历每个标注的框
                for bi in range(number_box):
                    bbox=batch_labels['boxes'][bi]
                    _,himg,wimg = batch[0][i].numpy().shape
                    bbox = bbox/ torch.tensor([wimg,himg,wimg,himg])
                    #import pdb
                    #pdb.set_trace()
                    center_x= (bbox[0]+bbox[2])*0.5
                    center_y= (bbox[1]+bbox[3])*0.5
                    #print("[%s,%s,%s],[%s,%s,%s]"%(wi/grid_number,center_x,(wi+1)/grid_number,hi/grid_number,center_y,(hi+1)/grid_number))
                    if center_x<=(wi+1)/grid_number and center_x>=wi/grid_number and center_y<=(hi+1)/grid_number and center_y>= hi/grid_number:
                        #pdb.set_trace()
                        cbbox =  torch.cat([torch.ones(1),bbox])
                        # 中心点落在grid内，
                        target_batch[i:i+1,wi:wi+1,hi:hi+1,:] = torch.unsqueeze(cbbox,0)
                    #else:
                        #cbbox =  torch.cat([torch.zeros(1),bbox])
                #import pdb
                #pdb.set_trace()
                #rint(target_batch[i:i+1,wi:wi+1,hi:hi+1,:])
                #target_batch[i:i+1,wi:wi+1,hi:hi+1,:] = torch.unsqueeze(cbbox,0)
    return target_batch
    

num_classes = 2
n_class    = 2
batch_size = 6
epochs     = 500
lr         = 1e-3
momentum   = 0
w_decay    = 1e-5
step_size  = 50
gamma      = 0.5

# 定义模型
yolov1_model = YOLOV1()
import pdb
pdb.set_trace()
# 定义优化算法为sdg:随机梯度下降
optimizer = optim.SGD(yolov1_model.detector.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)

# 定义学习率变化策略
# 每30个epoch 学习率乘以0.5
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

# 矩阵形式写法，写法简单，但是可读性不强
def lossfunc(outputs,labels):
    #import  pdb
    #pdb.set_trace()
    tmp = (outputs-labels)**2
    return torch.sum(tmp,0).view(1,5).mm(torch.tensor([10,0.0001,0.0001,0.0001,0.0001]).view(5,1))

# 定义直接拟合的学习率，可读性强
def lossfunc_details(outputs,labels):
    # 判断维度
    assert ( outputs.shape == labels.shape),"outputs shape[%s] not equal labels shape[%s]"%(outputs.shape,labels.shape)
    #import pdb
    #pdb.set_trace()
    b,w,h,c = outputs.shape
    loss = 0
    #import pdb
    #pdb.set_trace()
    conf_loss_matrix = torch.zeros(b,w,h)
    geo_loss_matrix = torch.zeros(b,w,h)
    loss_matrix = torch.zeros(b,w,h)
    
    for bi in range(b):
        for wi in range(w):
            for hi in range(h):
                #import pdb
                #pdb.set_trace()
                # detect_vector=[confidence,x,y,w,h]
                detect_vector = outputs[bi,wi,hi]
                gt_dv = labels[bi,wi,hi]
                conf_pred = detect_vector[0]
                conf_gt = gt_dv[0]
                x_pred = detect_vector[1]
                x_gt = gt_dv[1]
                y_pred = detect_vector[2]
                y_gt = gt_dv[2]
                w_pred = detect_vector[3]
                w_gt = gt_dv[3]
                h_pred = detect_vector[4]
                h_gt = gt_dv[4]
                loss_confidence = (conf_pred-conf_gt)**2 
                #loss_geo = (x_pred-x_gt)**2 + (y_pred-y_gt)**2 + (w_pred**0.5-w_gt**0.5)**2 + (h_pred**0.5-h_gt**0.5)**2
            
                loss_geo = (x_pred-x_gt)**2 + (y_pred-y_gt)**2 + (w_pred-w_gt)**2 + (h_pred-h_gt)**2
                loss_geo = conf_gt*loss_geo
                loss_tmp = loss_confidence + 0.3*loss_geo
                #print("loss[%s,%s] = %s,%s"%(wi,hi,loss_confidence.item(),loss_geo.item()))
                loss += loss_tmp
                conf_loss_matrix[bi,wi,hi]=loss_confidence
                geo_loss_matrix[bi,wi,hi]=loss_geo
                loss_matrix[bi,wi,hi]=loss_tmp
    #打印出batch中每张片的位置loss,和置信度输出
    print(geo_loss_matrix)
    print(outputs[0,:,:,0]>0.5)
    return loss,loss_matrix,geo_loss_matrix,conf_loss_matrix
# train
def train():
    for epoch in range(epochs):
        ts = time.time()
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()
            # 取图片
            inputs = input_process(batch)
            # 取标注
            labels = target_process(batch)
            
            # 获取得到输出
            outputs = yolov1_model(inputs)
            #import pdb
            #pdb.set_trace()
            #loss = criterion(outputs, labels)
            loss,lm,glm,clm = lossfunc_details(outputs,labels)
            loss.backward()
            optimizer.step()
            #print(torch.cat([outputs.detach().view(1,5),labels.view(1,5)],0).view(2,5))
            if iter % 10 == 0:
            #    print(torch.cat([outputs.detach().view(1,5),labels.view(1,5)],0).view(2,5))
                print("epoch{}, iter{}, loss: {}, lr: {}".format(epoch, iter, loss.data.item(),optimizer.state_dict()['param_groups'][0]['lr']))
        
        #print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        #print("*"*30)
        #val(epoch)
        scheduler.step()
# inference
def val(epoch):
    yolov1_model.eval()
    total_ious = []
    pixel_accs = []
    for iter, batch in enumerate(val_loader):
        inputs = input_process(batch)
        target,label= target_process(batch)

        output = yolov1_model(inputs)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
        

if __name__ == "__main__":
    train()
