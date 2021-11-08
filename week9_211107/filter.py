#coding:utf-8
import cv2
import numpy as np
import os
img  = cv2.imread("../week1/lena.jpg")
# # 图像滤波/卷积
kernel = np.ones((3,3),np.float32)/8
kernel=-kernel
kernel[0,:]=[-1,0,1]
kernel[1,:]=[-1,0,1]
kernel[2,:]=[-1,0,1]

print(kernel)
#plt.imshow(img)


# In[140]:


#dst=cv.filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]])；当ddepth=-1>    时，表示输出图像与原图像有相同的深度。
print(img.shape)
result = cv2.filter2D(img,-1,kernel)
result.shape
print(result[0,0])
#plt.imshow(result*255)


# In[118]:


result = cv2.filter2D(result,-1,kernel)
cv2.imwrite("filter2d_result.png",result)
os.system("open filter2d_result.png")
os.system("open ../week1/lena.jpg")
