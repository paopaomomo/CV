# encoding:utf-8
#
# created by xiongzihua
#
import torch
from torch.autograd import Variable
import torch.nn as nn
import sys
from resnet_yolo_v1 import resnet50
import torchvision.transforms as transforms
import cv2
import numpy as np
import pdb
import os

classes_list = ["今麦郎", "冰露", "百岁山", "怡宝", "百事可乐", "景甜", "娃哈哈", "康师傅", "苏打水", "天府可乐",
                "可口可乐", "农夫山泉", "恒大冰泉", "其他"]


def nms(bboxes, scores, threshold=0.5):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    # pdb.set_trace()
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    _, index = scores.sort(0, descending=True)
    print(index)
    order_array = index.numpy()
    print(order_array)
    index = order_array
    print(index)
    index_len = len(index)
    print(index_len)
    keep = []
    # while index.numel() > 0:
    while index_len > 0:
        index_len = index_len - 1
        i = index[0]  # every time the first is the biggst, and add it directly

        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap

        y11 = np.maximum(y1[i], y1[index[1:]])

        x22 = np.minimum(x2[i], x2[index[1:]])

        y22 = np.minimum(y2[i], y2[index[1:]])

        w_tmp = np.maximum(0, x22 - x11 + 1)  # the weights of overlap

        h_tmp = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w_tmp * h_tmp  # 重叠面积/交集面积

        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)  # 并集面积
        # pdb.set_trace()
        print("index[%d] ious=" % (i))
        print(ious)
        idx = np.where(ious <= threshold)[0]
        print(idx)
        index = index[idx + 1]  # because index start from 1

    print(keep)
    return torch.LongTensor(keep)

def decoder(pred):
    '''
    pred (tensor)
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    # pdb.set_trace()

    grid_num = 14
    boxes = []
    cls_indexs = []
    probs = []
    cell_size = 1. / grid_num
    pred = pred.data
    pred = pred.squeeze(0)  #
    contain1 = pred[:, :, 4].unsqueeze(2)
    contain2 = pred[:, :, 9].unsqueeze(2)
    contain = torch.cat((contain1, contain2), 2)
    mask1 = contain > 0.1  # 大于阈值
    mask2 = (contain == contain.max())  # we always select the best contain_prob what ever it>0.9pred[:,:,4]
    mask = (mask1 + mask2).gt(0)
    # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                # print("b=%d" %(b))
                # index = min_index[i,j]
                # mask[i,j,index] = 0
                if mask[i, j, b] == 1:
                    print("mask[%d,%d,%d]==1" % (i, j, b))
                    # print(i,j,b)
                    box = pred[i, j, b * 5:b * 5 + 4]
                    contain_prob = torch.FloatTensor([pred[i, j, b * 5 + 4]])
                    xy = torch.FloatTensor([j, i]) * cell_size  # cell左上角  up left of cell
                    box[:2] = box[:2] * cell_size + xy  # return cxcy relative to image
                    box_xy = torch.FloatTensor(box.size())  # 转换成xy形式    convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]
                    max_prob, cls_index = torch.max(pred[i, j, 10:], 0)

                    print("get a box")
                    print(box_xy)
                    print("the box cls=")
                    print(cls_index)
                    print("the box max_prob")
                    print(max_prob)
                    print("the box contain_prob")
                    print(contain_prob)
                    print("float((contain_prob*max_prob)[0])")
                    print(float((contain_prob * max_prob)[0]))
                    ENABLE_VALUE = 0.1
                    # ENABLE_VALUE = 0.02

                    # if float((contain_prob*max_prob)[0]) > ENABLE_VALUE:
                    if float((contain_prob * max_prob)[0]) > ENABLE_VALUE:

                        print("find a box (%d %d %d)" % (i, j, b))
                        boxes.append(box_xy.view(1, 4))
                        # pdb.set_trace()

                        tmp_list = []
                        tmp_int = cls_index.item()
                        tmp_list.append(tmp_int)
                        tmp_tensor = torch.tensor(tmp_list)
                        cls_indexs.append(tmp_tensor)
                        # cls_indexs.append(cls_index)

                        # cls_indexs.append(float(cls_index.item()))

                        probs.append(contain_prob * max_prob)
                    else:
                        print("contain_prob*max_prob not > 0.1")
                # else:
                # print("mask not 1")
    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        # pdb.set_trace()
        boxes = torch.cat(boxes, 0)  # (n,4)
        probs = torch.cat(probs, 0)  # (n,)
        # pdb.set_trace()
        # print(cls_indexs)
        cls_indexs_len = len(cls_indexs)
        print("cls_indexs_len=%d" % (cls_indexs_len))

        # cls_indexs = torch.cat(torch.tensor(cls_indexs),0) #(n,)
        cls_indexs = torch.cat(cls_indexs, 0)  # (n,)

    keep = nms(boxes, probs)
    return boxes[keep], cls_indexs[keep], probs[keep]

def predict(model, image_name, root_path=''):
    result = []
    print("img name=%s" % (image_name))
    image = cv2.imread(image_name)

    h, w, _ = image.shape
    img = cv2.resize(image, (448, 448))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = (141, 132, 126)  # RGB
    img = img - np.array(mean, dtype=np.float32)

    transform = transforms.Compose([transforms.ToTensor(), ])
    img = transform(img)
    img = Variable(img[None, :, :, :], volatile=True)
    img = img

    pred = model(img)
    pred = pred.cpu()


    # return (tensor) box[[x1,y1,x2,y2]] label[...]
    boxes, cls_indexs, probs = decoder(pred)

    for i, box in enumerate(boxes):
        x1 = int(box[0] * w)
        x2 = int(box[2] * w)
        y1 = int(box[1] * h)
        y2 = int(box[3] * h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index)  # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1, y1), (x2, y2), classes_list[cls_index], image_name, prob])
    return result


def write_output_file(result):
    i = 0
    for left_up, right_bottom, class_name, _, prob in result:
        # pdb.set_trace()
        print("result index=%d" % (i))
        print(result)
        if i == 0:
            tmp_len = len(_.split('/'))
            file_name = _.split('/')[tmp_len - 1]
            file_name = file_name.split('.')[0]
            output_predict_txt = output_file_dir + file_name + '.txt'
            print("############## write file %s##############" % (output_predict_txt))

            output_predict_file = open(output_predict_txt, 'w')

            output_predict_file.write(
                str(class_name) +
                ' ' +
                str(prob) +
                ' ' +
                str(left_up[0]) +
                ' ' +
                str(left_up[1]) +
                ' ' +
                str(right_bottom[0]) +
                ' ' +
                str(right_bottom[1])
            )
            output_predict_file.write('\n')

        i = i + 1

if __name__ == '__main__':
    pass






