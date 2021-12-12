# encoding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

yolo_v1_output = 24  # 5*2 + 14


class yoloLoss(nn.Module):  # (7, 2, 5, 0.5)
    def __init__(self, S, B, l_coord, l_noobj):
        # 为了更重视8维的坐标预测，给这些算是前面赋予更大的loss weight
        # 对于有物体的记为λ,coord，在pascal VOC训练中取5，
        # 对于没有object的bbox的confidence loss，前面赋予更小的loss weight 记为 λ,noobj, 在pascal VOC训练中取0.5
        # 有object的bbox的confidence loss (上图红色框) 和类别的loss （上图紫色框）的loss weight正常取1
        super(yoloLoss, self).__init__()
        self.S = S  # 7
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):
        # iou 是求两个框的交并比
        # iou可用于求loss,可测试时的NMS

        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self, pred_tensor, target_tensor):  # s-size=14 B-boxcount=2
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,30)
        '''
        # print('pred_tensor.size(): ', pred_tensor.size())
        # print('target_tensor.size(): ', target_tensor.size())
        import pdb
        pdb.set_trace()
        N = pred_tensor.size()[0]  # batch-size N=3
        coo_mask = target_tensor[:, :, :, 4] > 0  # 具有目标标签的索引
        noo_mask = target_tensor[:, :, :, 4] == 0  # 不具有目标的标签索引
        # 得到含物体的坐标等信息
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        coo_mask = coo_mask.bool()
        # 得到不含物体的坐标等信息
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.bool()

        coo_pred = pred_tensor[coo_mask].view(-1, yolo_v1_output)
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5)  # box[x1,y1,w1,h1,c1]
        class_pred = coo_pred[:, 10:]  # [x2,y2,w2,h2,c2]
        import pdb
        pdb.set_trace()
        coo_target = target_tensor[coo_mask].view(-1, yolo_v1_output)
        box_target = coo_target[:, :10].contiguous().view(-1, 5)
        class_target = coo_target[:, 10:]

        # 1. compute not contain obj loss
        noo_pred = pred_tensor[noo_mask].view(-1, yolo_v1_output)
        noo_target = target_tensor[noo_mask].view(-1, yolo_v1_output)
        noo_pred_mask = torch.ByteTensor(noo_pred.size())
        noo_pred_mask.zero_()
        noo_pred_mask[:, 4] = 1
        noo_pred_mask[:, 9] = 1
        noo_pred_mask = noo_pred_mask.bool()
        noo_pred_c = noo_pred[noo_pred_mask]  # noo pred只需要计算 c 的损失 size[-1,2]
        noo_target_c = noo_target[noo_pred_mask]
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, size_average=False)  # 对应的位置做均方误差

        # compute contain obj loss
        coo_response_mask = torch.ByteTensor(box_target.size())
        coo_response_mask.zero_()
        coo_not_response_mask = torch.ByteTensor(box_target.size())
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size())

        # 预测值，有多个box的话那么就取一个最大的box，出来就可以了其他的不要啦
        for i in range(0, box_target.size()[0], 2):  # choose the best iou box
            # choose the best iou box ， box1 是预测的 box2 是我们提供的
            import pdb
            pdb.set_trace()
            box1 = box_pred[i:i + 2]
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            # box1 : tx,ty,w,h
            box1_xyxy[:, :2] = box1[:, :2] / 14. - 0.5 * box1[:, 2:4]
            box1_xyxy[:, 2:4] = box1[:, :2] / 14. + 0.5 * box1[:, 2:4]
            box2 = box_target[i].view(-1, 5)
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:, :2] = box2[:, :2] / 14. - 0.5 * box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2] / 14. + 0.5 * box2[:, 2:4]
            iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])  # [2,1]
            max_iou, max_index = iou.max(0)
            max_index = max_index.data

            coo_response_mask[i + max_index] = 1
            coo_not_response_mask[i + 1 - max_index] = 1

            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            box_target_iou[i + max_index, torch.LongTensor([4])] = (max_iou).data

        # pdb.set_trace()
        box_target_iou = Variable(box_target_iou)

        # 2.response loss
        # 2.response loss，iou符合的
        box_pred_response = box_pred[coo_response_mask].view(-1, 5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1, 5)
        box_target_response = box_target[coo_response_mask].view(-1, 5)
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], size_average=False)

        # 3.loc_loss
        center_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], size_average=False)
        print("center_loss= ", center_loss.item())


        hw_loss = F.mse_loss(box_pred_response[:, 2:4], box_target_response[:, 2:4], size_average=False)

        print("hw_loss= ", hw_loss.item())

        loc_loss = center_loss + hw_loss
        # print("loc_loss= ", loc_loss.item())

        # 4.not response loss
        # 3.not response loss iou不符合的
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1, 5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1, 5)
        box_target_not_response[:, 4] = 0
        # not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)
        # I believe this bug is simply a typo
        not_contain_loss = F.mse_loss(box_pred_not_response[:, 4], box_target_not_response[:, 4], size_average=False)

        # 5.class loss
        class_loss = F.mse_loss(class_pred, class_target, size_average=False)
        #
        print("l_coord= ", self.l_coord)  # 5
        print("loc_loss= ", loc_loss.item())  #
        print("contain_loss= ", contain_loss.item())  #
        print("not_contain_loss= ", not_contain_loss.item())  #
        print("nooobj_loss= ", nooobj_loss.item())  # 342.2611
        print("class_loss= ", class_loss.item())  # 3.4275

        all_loss = (self.l_coord * loc_loss + 2 * contain_loss + not_contain_loss + self.l_noobj * nooobj_loss + class_loss) / N
        #
        print("return loss= ", all_loss.item())
        return all_loss










