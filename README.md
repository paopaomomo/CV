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
