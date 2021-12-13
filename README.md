# CCV10's CVFundamentals
  - computer vision fundamentals 
  - 以下文件都可以在https://gitee.com/anjiang2020_admin/ccv10     找到，若找不到联系，可以明明老师协助。微信13271929138
  - week7 
```
CV核心基础WEEK7 :  基本图像处理
https://gitee.com/anjiang2020_admin/ccv10
Pipeline:
0.职业规划
1.Computer Vision 的由来
2.计算机如何看到图像
3.计算机处理图像的方式，方法

作业：
   1 用滤波操作给图片去除噪声，
     选做：将自己的logo水印打到经过滤波后的照片上。
   目的：感受滤波操作的作用，用数字上的滤波操作模拟老照片真实镜头的滤波功能。
   参考步骤：
   1 拿到老师给定的图片：week7_211017/week7_homework.png。
   2 对图片进行滤波操作。参考week7_211017/week7_class_code_after_class.py
   3 修改滤波核的数值和滤波核的大小，调整出最好的效果。
   4 制作自己的logo水印的照片
   5 将水印添加到图片上。参考week7_211017/week7_class_code_after_class.py
   6 图像围绕任意点进行刚体旋转的公式推导：https://zhuanlan.zhihu.com/p/399377684
```
 - week8 
```
 CV核心基础WEEK8 ：认识计算机视觉[https://gitee.com/anjiang2020_admin/ccv10]
 Pipeline:
 1.    图像处理与计算机视觉
 2.    计算机视觉的输入与输出
 3.    如何解决计算机视觉的几个问题
 4.    计算机视觉第一步：图像描述子

 作业：
  
  编写计算机视觉的第0版程序。
  步骤
  1 生成10张图片，对应0,1,2,3,4,5,6,7,8,9.
  2 对这10张图片提取特征x。
  3 用一个判别器f(x)来决策输出结果y。
    这个判别器达到作用：
    当x是 “0”图片对应的特征时，y=f(x)=0
    当x是 “1”图片对应的特征时，y=f(x)=1
    当x是 “2”图片对应的特征时，y=f(x)=2
    当x是 “3”图片对应的特征时，y=f(x)=3
    当x是 “4”图片对应的特征时，y=f(x)=4
    当x是 “5”图片对应的特征时，y=f(x)=5
    当x是 “6”图片对应的特征时，y=f(x)=6
    当x是 “7”图片对应的特征时，y=f(x)=7
    当x是 “8”图片对应的特征时，y=f(x)=8
    当x是 “9”图片对应的特征时，y=f(x)=9
 4 参考代码:week2/recognize_computer_vision.py
 
 课堂参考资料：
   1. 灰度变换，gamma变换的例子：gamma.py
   2. 课后作业参考：recognize_computer_vision.py
   3. 第一周课后作业参考：week7_homework.ipynb
   4. 第一周课后作业参考：week7_homework.py
   5. week2课堂代码：week8_class_code.ipynb
   6. week2课堂代码: week8_class_code.py
   7. 课堂用图：tangsan.jpg
   8. 课堂用图：dog.png
   9. 关键点算法：https://zhuanlan.zhihu.com/p/147390611
```
 - week9 
```
CV核心基础WEEK9 ：依赖硬件算力提升模型性能：cuda编程
https://gitee.com/anjiang2020_admin/ccv10
Pipeline:
1  week8作业 
2  计算机视觉的常用模型
3  CNN统一了提特征与决策
4  GPU Schema
5  认识pycuda,并用pycuda完成矩阵乘法

作业：使用pycuda完成LeNet模型的以下模块：
        1. [必做]conv
        2. pooling
        3. relu
        4. linear
        5. backward

要求：
   1 要求用pycuda库利用gpu的多线程技术，完成卷积层的计算。
   2 可以用自己定义的kernel函数,也可以用pycuda提供的核函数
   3 自己定义核函数的时候，可以参考week9_pycuda_example_6.py来实现
其它参考材料：
    1. week8参考作业：week9_20210313/week8作业答案课堂讲解.ipynb
    2. 卷积层用nn.conv2d来实现，相关参考代码：week6/conv.py
    3. 卷积的声明，更改默认weight，默认bias,对图片进行卷积，示例子代码在：week9_20210313/conv.py 的14行,27行,29行,55行
    4. 图像滤波器：filter.py
    5. pycuda-master: pycuda源码
    6. week9_pe_5.py : 自定义核函数，打印出“hello world"
    7. week9_pe_4.py : 自定义核函数，掌握threadIdx,blockIdx等内置变量的意义。
    8. week9_pycuda_example_2.py ： 自定义加法核函数
    9. week9_pycuda_example_3.py :  自定义乘法核函数
    10. week9_pycuda_example_6.py :  自定义矩阵乘法核函数
    11. week9_pycude_example_1.py :  利用gpuarray来调用gpu进行计算。
    12. week9_cvf.py : pycuda 自带api使用,gpuarray,以及自带核函数的使用
```
- week10 [图像分类的决策层设计总结与实战]
```
https://gitee.com/anjiang2020_admin/ccv10
CV核心基础WEEK10：图像分类决策层的设计总结与实践
Pipeline:
1  CNN提特征层的设计：搭积木 
2  看看经典模型是如何搭积木的
3  决策功能的实现：output层的设计
4  生成output需要的groundtruth
5  经典模型Resnet/mobilenet

作业：自己完成一个分类项目：数据采集，标注，设计网络（可从头开始，也可fineturn).
       1. [可选]每人提交10张矿泉水瓶\可乐瓶的图片到邮箱：471106585qq.com
            ![输入图片说明](https://images.gitee.com/uploads/images/2020/0715/135022_28d7b639_7401441.png "屏幕截图.png")
             ![输入图片说明](https://images.gitee.com/uploads/images/2020/0715/135042_997cd525_7401441.png "屏幕截图.png")
       2. 这次分类的图片由老师标注。统一传到modelarts标注平台标注。[检测类的项目时，老师会将图片统一传到modelarts标注平台，交给大家标注]
       3. 自行完成分类网络的决策层。体特征层可从零设计，也可使用其他网络。建议resnet18,在week10_210902/resnet.py中有其实现，可参考
       4. 此次数据集明名为week10_dataset,永远开放的学习型数据集,数据的data_loader也在最近几天提供在week10_dataset的gitee里:https://gitee.com/anjiang2020_admin/week10-dataset

要求：
   1 提交作业时，需要提交代码，训练的train acc
   2 [可选] 提交训练超参数（学习率策略，优化方法，优化epoch数，train acc,test acc)
  
其它参考材料：
    1. mobilenets论文：mobilenets_paper.pdf
    2. week9作业参考答案：pycuda_conv_week9_homework.py
    3. resnet网络搭建代码参考：resnet.py
    4. pytorch实现alexnet,代码在文件夹week10_210902/alexnet下，实现细节可参考:http://www.imlarger.com:8081/dist/index.html#/Paper/267 如果此网址打开困难，可以看week10_210902/alexnet/训练一个alexnet.pdf

    5. week10_dataset：:https://gitee.com/anjiang2020_admin/week10-dataset
```

- week11 [图像检测决策层设计总结与实战(一)]
```
[ https://gitee.com/anjiang2020_admin/ccv10 ]
CV核心基础WEEK11：一阶段的图像检测模型的决策层设计总结与实战
Pipeline:
1  只用计算一次就能到得到检测框:YOLO 
2  多尺度提取特征:YOLOV2


作业：1 下载提供的week10-datasets-detect 数据集:https://gitee.com/anjiang2020_admin/week10-dataset
     2 在分类模型的基础上，加上检测层，对week10-datasets进行回归检测
     说明：
       1. 自行完成检测层的设计。自行决定：检测头的检测区域个数，每个检测区域内输出框数，类别数。
       2. 网络得backbone可以是你week10使用得网络，也可以与加载一些经典网络，也可以自行设计。
       3. NMS得实现可参考ppt.代码可自己写，也可以从网站找，网上代码一般含有IoU.
       
要求：
   1 提交作业时，需要提交代码，训练超参数（学习率策略，优化方法，优化epoch数，给出10张图片以及其对应检测结果。)
  
其它参考材料：
    1. week11_210327/矿泉水瓶分类/bcnn_iccv15.pdf 
    2. 矿泉水瓶分类.ipynb 
    3. YOLO_V1论文：YOLO_V1.pdf 
    4. YOLO_V2论文：YOLO_V2.pdf 
    5. YOLO_V3论文：YOLO_V3.pdf 
    6. [推荐]YOLO至YOLOv5通俗易懂文字教程：https://zhuanlan.zhihu.com/p/183261974
    7. 2016CVPR会议上，作者的PPT:YOLO_CVPR_2016_ppt.pdf
    8. [重要，建议背诵]微调模型示例：Finetuning_convnet.py Finetuning_convnet.ipynb  

```
week12 [YOLO之前：图像检测决策层设计总结与实战(二)]
```
[ https://gitee.com/anjiang2020_admin/ccv10 ]
CV核心基础WEEK12：YOLO之前：图像检测决策层设计总结与实战(二)
Pipeline:
1  如何评价检测器性能？ 
2  深度学习初次用于检测：RCNN
3  比CRNN快213倍：Fast RCNN
4  真正得端到端：Faster RCNN 

作业：
     1. 计算week11所设计检测器的mAP，先把每个子类别得AP算出来，然后计算mAP。如果把所有水瓶算做一类，就算一个AP出来即可。
     2. [faster rcnn学完之后选做]用region based得方法【rpn】,可能会更好  
要求：
   1. 提交作业时，需要与自己检测器相适配得mAP代码，以及总得mAP值，分类别得AP值。
  
其它参考材料：
   1. mAP 代码可参考https://github.com/Cartucho/mAP 
   2. week11作业参考中的pennfudan数据下载地址:https://www.cis.upenn.edu/~jshi/ped_html/
   3. RCNN模型后续系列的发展，可以参考：https://zhuanlan.zhihu.com/p/368483790，比如这里有个：双头RCNN
   
```
week13 [分割网络的设计]
```
[ https://gitee.com/anjiang2020_admin/ccv10 ]
CV核心基础WEEK13：分割网络的设计
Pipeline:
1  分割器的设计思路
2  经典分割模型的涨点方法
 

作业：对老师提供的FCN代码填空，对图像进行分割
FCN 参考步骤：
     1. 编写网络结构文件week13_211205/homework/fcn.py
          18行，补齐FCN32s网络各层得定义：
          [可选] 52行，补齐FCN8s各层得定义
          [可选] 71行，实现跳级结构

       2. 待准备pennfudan数据集https://www.cis.upenn.edu/~jshi/ped_html/
       所以，在一开始我们掌握模型原理阶段，需要一个小数据集
       pennfudan是一个只有52M的小型数据集，我们就用它的验证网络的有效性。尽快掌握FCN
       pennfudan数据下载地址:https://www.cis.upenn.edu/~jshi/ped_html/
       ![输入图片说明](https://images.gitee.com/uploads/images/2020/0706/195021_9b419532_7401441.png "屏幕截图.png")
       ![输入图片说明](https://images.gitee.com/uploads/images/2020/0706/200412_b841b066_7401441.png "屏幕截图.png")

       3. 预先训练好得模型:week13_211205/homework/models/文件夹下

选做作业: 参考FCN,deeplab等，设计一种检测器，对week10_dataset中的瓶子进行分割。
建议步骤: 
     1. week10_dataset标注，标注文件在https://gitee.com/anjiang2020_admin/week10-dataset
     2. dataloader的编写
     3. 按照Deeplab的思路，重新设计一个网络
```
week14- [跟踪算法的设计]

```
CV核心基础WEEK14 :  计算机视觉之跟踪算法的设计
[ https://gitee.com/anjiang2020_admin/ccv10 ]
Pipeline:
0.week13homework
1.跟踪任务与检测任务的区别
2.用分类做跟踪
3. MeanShift做跟踪
4. 相关滤波方法做跟踪
5. 深度学习做跟踪

作业：
   1 写一个tracker，跟踪水瓶（老师提供视频数据【已标注】：https://gitee.com/anjiang2020_admin/week10-dataset/blob/master/README.md#week14跟踪数据集）
     选做：MOSSE方法或者learning to track 100FPS using DNN。
   目的：掌握基于分类做跟踪的思路；掌握跟踪算法中，在线更新模型的的操作办法。
   其它文件：
        week13作业答案：week14/week13_homework_answer/answer/
        yolo的另一种实现：week14/CV_homework_week12
        week14/将相关滤波器跟踪算法的速度做到极致.pdf 
   待跟踪视频：https://www.bilibili.com/video/BV1Y54y1273C/
   ```
   