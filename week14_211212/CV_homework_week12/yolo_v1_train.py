#coding:utf-8
import numpy as np
from Week10Dataset_process import myDataset

from torch.autograd import Variable
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from resnet_yolo_v1 import resnet50
from yolo_v1_loss import yoloLoss
from predict_decoder import decoder
import cv2
import torch




device = 'cuda' if torch.cuda.is_available() else 'cpu'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

classes_list = ["今麦郎", "冰露", "百岁山", "怡宝", "百事可乐", "景甜", "娃哈哈", "康师傅", "苏打水", "天府可乐",
                "可口可乐", "农夫山泉", "恒大冰泉", "其他"]
Color = [[0, 0, 0],
         [128, 0, 0],
         [0, 128, 0],
         [128, 128, 0],
         [0, 0, 128],
         [128, 0, 128],
         [0, 128, 128],
         [128, 128, 128],
         [64, 0, 0],
         [192, 0, 0],
         [64, 128, 0],
         [192, 128, 0],
         [64, 0, 128],
         [192, 0, 128],
         [64, 128, 128],
         [192, 128, 128],
         [0, 64, 0],
         [128, 64, 0],
         [0, 192, 0],
         [128, 192, 0],
         [0, 64, 128]]


img_path = 'week10_dataset/image/'
batch_size = 2
learning_rate = 0.002
num_epochs = 1


# train dataset
train_dataset = myDataset(img_path=img_path, file_name='train.txt', train=True, transform=[transforms.ToTensor()])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# test dataset
test_dataset = myDataset(img_path=img_path, file_name='val.txt', train=False, transform=[transforms.ToTensor()])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# network structure
net = resnet50()
net = net.to(device)
# load pre_trained model
resnet = models.resnet50(pretrained=True)
new_state_dict = resnet.state_dict()
#

op = net.state_dict()
for k in new_state_dict.keys():
    print(k)
    if k in op.keys() and not k.startswith('fc'):  # startswith() 方法用于检查字符串是否是以指定子字符串开头，如果是则返回 True，否则返回 False
        print('yes')
        op[k] = new_state_dict[k]
net.load_state_dict(op)

criterion = yoloLoss(7, 2, 5, 0.5)

net.train()
# different learning rate
params = []
params_dict = dict(net.named_parameters())

for key, value in params_dict.items():
    if key.startswith('features'):
        params += [{'params': [value], 'lr':learning_rate * 1}]
    else:
        params += [{'params': [value], 'lr':learning_rate}]

optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

torch.multiprocessing.freeze_support()

for epoch in range(num_epochs):
    net.train()
    if epoch == 30:
        learning_rate = 0.0001
    if epoch == 40:
        learning_rate = 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))

    total_loss = 0.0
    for i, (images, target) in enumerate(train_loader):
        # images, target = images.cuda(), target.cuda() # use gpu
        pred = net(images)
        print('pred.size(): ', pred.size())
        print('target.size(): ', target.size())
        loss = criterion(pred, target)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 5 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' % (epoch + 1, num_epochs,
                                                                                      i + 1, len(train_loader),
                                                                                      loss.item(),
                                                                                      total_loss / (i + 1)))


# predict
val_file_name = 'val.txt'
result = []
file_path = os.path.join(img_path, val_file_name)
with open(file_path) as f:
    lines = f.readlines()  # xxx.jpg xx xx xx xx class
    box = []
    label = []
    for line in lines:
        splited = line.strip().split()
        img_name = splited[0]  # 存储图片的地址+图片名称
        image = cv2.imread(img_name)
        h, w, _ = image.shape
        img = cv2.resize(image, (448, 448))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mean = (141, 132, 126)  # RGB
        img = img - np.array(mean, dtype=np.float32)
    
        transform = transforms.Compose([transforms.ToTensor(), ])
        img = transform(img)
        img = Variable(img[None, :, :, :], volatile=True)
        #             img = img.cuda()
    
        pred = net(img)  # 1x14x14x24
        pred = pred.cpu()
    
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
            result.append([(x1, y1), (x2, y2), classes_list[cls_index], img_name, prob])



for left_up, right_bottom, class_name, img_name, prob in result:
    image = cv2.imread(img_name)
    output_name = class_name
    color = Color[classes_list.index(class_name)]
    cv2.rectangle(image, left_up, right_bottom, color, 2)
    label = class_name + str(round(prob, 2))
    text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    p1 = (left_up[0], left_up[1] - text_size[1])
    cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]),
                  color, -1)
    cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)
    # cv2.imshow(output_name, image)


    # validation_loss = 0.0
    #     # best_test_loss = np.inf
    #     # net.eval()
    #     #
    #     # for i, (images, target) in enumerate(test_loader):
    #     #     # images, target = images.cuda(), target.cuda()
    #     #     pred = net(images)
    #     #     loss = criterion(pred, target)
    #     #     validation_loss += loss.item()
    #     # validation_loss /= len(test_loader)


    # if best_test_loss > validation_loss:
    #     best_test_loss = validation_loss
    #     print('get best test loss %.5f' % best_test_loss)
    #     torch.save(net.state_dict(), 'best.pth')
    # torch.save(net.state_dict(), 'yolo.pth')

