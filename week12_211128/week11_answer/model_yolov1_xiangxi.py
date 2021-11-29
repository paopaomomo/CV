import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import cv2




class VGG(nn.Module):
    def __init__(self):
       super(VGG,self).__init__()
       # the vgg's layers
       #self.features = features
       cfg = [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']
       # use the vgg layers to get the feature
       #import pdb
       #pdb.set_trace()
       self.conv2d_1 = nn.Conv2d(3,64,3,1)
       self.bn_1 = nn.BatchNorm2d(64)
       self.relu_1 = nn.ReLU(True)
       
       self.conv2d_2 = nn.Conv2d(64,64,3,1)
       self.bn_2 = nn.BatchNorm2d(64)
       self.relu_2 = nn.ReLU(True)
       
       self.pool_2 = nn.MaxPool2d(2,2)
       self.conv2d_3 = nn.Conv2d(64,128,3,1)
       self.bn_3 = nn.BatchNorm2d(128)
       self.relu_3 = nn.ReLU(True)
       
       self.conv2d_4 = nn.Conv2d(128,128,3,1)
       self.bn_4 = nn.BatchNorm2d(128)
       self.relu_4 = nn.ReLU(True)
       
       self.pool_4 = nn.MaxPool2d(2,2)
       self.conv2d_5 = nn.Conv2d(128,256,3,1)
       self.bn_5 = nn.BatchNorm2d(256)
       self.relu_5 = nn.ReLU(True)
       
       self.conv2d_6 = nn.Conv2d(256,256,3,1)
       self.bn_6 = nn.BatchNorm2d(256)
       self.relu_6 = nn.ReLU(True)
       
       self.conv2d_7 = nn.Conv2d(256,256,3,1)
       self.bn_7 = nn.BatchNorm2d(256)
       self.relu_7 = nn.ReLU(True)
       
       self.pool_7 = nn.MaxPool2d(2,2)
       self.conv2d_8 = nn.Conv2d(256,512,3,1)
       self.bn_8 = nn.BatchNorm2d(512)
       self.relu_8 = nn.ReLU(True)
       
       self.conv2d_9 = nn.Conv2d(512,512,3,1)
       self.bn_9 = nn.BatchNorm2d(512)
       self.relu_9 = nn.ReLU(True)
       
       self.conv2d_10 = nn.Conv2d(512,512,3,1)
       self.bn_10 = nn.BatchNorm2d(512)
       self.relu_10 = nn.ReLU(True)
       
       self.pool_10 = nn.MaxPool2d(2,2)
       self.conv2d_11 = nn.Conv2d(512,512,3,1)
       self.bn_11 = nn.BatchNorm2d(512)
       self.relu_11 = nn.ReLU(True)
       
       self.conv2d_12 = nn.Conv2d(512,512,3,1)
       self.bn_12 = nn.BatchNorm2d(512)
       self.relu_12 = nn.ReLU(True)
       
       self.conv2d_13 = nn.Conv2d(512,512,3,1)
       self.bn_13 = nn.BatchNorm2d(512)
       self.relu_13 = nn.ReLU(True)
       
       self.pool_13 = nn.MaxPool2d(2,2)
       # 全局池化
       self.avgpool = nn.AdaptiveAvgPool2d((7,7))
       # 决策层：分类层
       self.classifier = nn.Sequential(
           nn.Linear(512*7*7,4096),
           nn.ReLU(True),
           nn.Dropout(),
           nn.Linear(4096,4096),
           nn.ReLU(True),
           nn.Dropout(),
           nn.Linear(4096,1000),
       )

       for m in self.modules():
           if isinstance(m,nn.Conv2d):
               nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
               if m.bias is not None: 
                   nn.init.constant_(m.bias,0)
           elif isinstance(m,nn.BatchNorm2d):
               nn.init.constant_(m.weight,1)
               nn.init.constant_(m.bias,1)
           elif isinstance(m,nn.Linear):
               nn.init.normal_(m.weight,0,0.01)
               nn.init.constant_(m.bias,0)

    def forward(self,x):
         x_records=[]
         #x = self.features(x)
         x = self.conv2d_1(x)
         x_records.append(x)
         x = self.bn_1(x)
         x = self.relu_1(x)
         x_records.append(x)
         x = self.conv2d_2(x)
         x = self.bn_2(x)
         x = self.relu_2(x)
         x_records.append(x)
         x = self.pool_2(x)
         x = self.conv2d_3(x)
         x = self.bn_3(x)
         x = self.relu_3(x)
         x_records.append(x)
         x = self.conv2d_4(x)
         x = self.bn_4(x)
         x = self.relu_4(x)
         x_records.append(x)

         x = self.pool_4(x)
         x = self.conv2d_5(x)
         x = self.bn_5(x)
         x = self.relu_5(x)
         x_records.append(x)
         x = self.conv2d_6(x)
         x = self.bn_6(x)
         x = self.relu_6(x)
         x_records.append(x)
         x = self.conv2d_7(x)
         x = self.bn_7(x)
         x = self.relu_7(x)
         x_records.append(x)
         x = self.pool_7(x)
         x = self.conv2d_8(x)
         x = self.bn_8(x)
         x = self.relu_8(x)
         x_records.append(x)
         x = self.conv2d_9(x)
         x = self.bn_9(x)
         x = self.relu_9(x)
         x_records.append(x)
         x = self.conv2d_10(x)
         x = self.bn_10(x)
         x = self.relu_10(x)
         x = self.pool_10(x)
         x = self.conv2d_11(x)
         x = self.bn_11(x)
         x = self.relu_11(x)
         x = self.conv2d_12(x)
         x = self.bn_12(x)
         x = self.relu_12(x)
         x = self.conv2d_13(x)
         x = self.bn_13(x)
         x = self.relu_13(x)
         x = self.pool_13(x)
         x_records.append(x)
         x = self.avgpool(x)
         x_records.append(x)
         x = x.view(x.size(0),-1)
         x = self.classifier(x)
         return x,x_records


if __name__ == '__main__':
    vgg = VGG()
    x  = torch.randn(1,3,512,512)
    #img=torch.tensor(cv2.imread("../../model_test.png"))
    img=torch.tensor(cv2.imread("../../model_test2.png"))
    w,h,c = img.shape
    x = img.view(1,w,h,c).permute(0,3,1,2).contiguous()
    x = x/255.
    import pdb
    pdb.set_trace()
    feature,x_records = vgg(x)
    print(feature.shape)
    from show_featuremap import show_featuremap
    for fea in x_records:
        print(fea.shape)
        show_featuremap(fea.detach())
    
