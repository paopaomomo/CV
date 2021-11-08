#coding:utf-8
import torch
from torch import nn
import os



# 生成tensor
x = torch.tensor([[1.,2.,3.,7.],[6,3,4,2]])
# 也可以从图片中读取
print(x)
import pdb
pdb.set_trace()
convolution=nn.Conv2d(1,1,kernel_size = 2,stride=2,padding=0)
#print(convolution(x))


# 查看卷积类都有哪些成员函数与成员变量
dir(convolution)
# 对x 进行维度变换
x1=x.view(1,1,2,4)
#with torch.no_grad()
print(convolution(x1))


#改变卷积核的取值
convolution.weight.data=torch.tensor([[[[1.,-1.],[1.,-1.]]]])
#改变偏移量得取值
convolution.bias.data =torch.tensor([0.0])
print(convolution(x1))

# 读取mnist数据
import sys
sys.path.append("../week4")
from mnist import MNIST
mndata = MNIST('../week4/mnist/python-mnist/data/')
image_data_all, image_label_all = mndata.load_training()
image_data=image_data_all[0:100]
image_label=image_label_all[0:100]
pdb.set_trace()

x = torch.tensor(image_data[0])
#np.array(image_data[3]).reshape((28,28)).astype('uint8')
img=x.view(1,1,28,28)

img_show=x.view(28,28)
import cv2
cv2.imwrite("img_show.png",img_show.numpy().astype("uint8"))
os.system("open img_show.png")
pdb.set_trace()



img = img.float()
feature= convolution(img)
feature_show =feature.view(15,15)
cv2.imwrite("feature_show.png",feature_show.detach().numpy().astype("uint8"))
os.system("open feature_show.png")
pdb.set_trace()
print(image_label)



def show_image(img):
    cv2.imwrite("tmp.png",img.detach().numpy().astype("uint8"))
    os.system("open tmp.png")

show_image(feature_show)
show_image(img_show)

show_image((convolution(torch.tensor(image_data[0]).view(1,1,28,28).float())).view(15,15))
conv2=nn.Conv2d(1,1,kernel_size = 2,stride=2,padding=1)
show_image((conv2(torch.tensor(image_data[0]).view(1,1,28,28).float())).view(15,15))


pdb.set_trace()
print("end")

