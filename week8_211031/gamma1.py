import cv2
import sys
import os


img=cv2.imread("tangsan.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


#伽马变换
gamma=gray.copy()
rows=img.shape[0]
cols=img.shape[1]
for i in range(rows):
    for j in range(cols):
        gamma[i][j]=6*pow(gamma[i][j],0.5)

cv2.imwrite("tangsan_gray_gamma.jpg",cv2.hconcat([gray,gamma]))
os.system("open tangsan_gray_gamma.jpg")

#import pdb
#pdb.set_trace()
#伽马变换
gamma_c=img.copy()
rows=img.shape[0]
cols=img.shape[1]
deeps=img.shape[2]
for i in range(rows):
    for j in range(cols):
        for d in range(deeps):
            gamma_c[i][j][d]=3*pow(gamma_c[i][j][d],0.8)

cv2.imwrite("tangsan_img_gamma_c.jpg",cv2.hconcat([img,gamma_c]))
os.system("open tangsan_img_gamma_c.jpg")


#反色变换
reverse=gray.copy()
rows=img.shape[0]
cols=img.shape[1]
for i in range(rows):
    for j in range(cols):
        reverse[i][j]=255-reverse[i][j]

cv2.imwrite("tangsan_gray_reverse.jpg",cv2.hconcat([gray,reverse]))
os.system("open tangsan_gray_reverse.jpg")

#反色变换
reverse_c=img.copy()
rows=img.shape[0]
cols=img.shape[1]
deeps=img.shape[2]
for i in range(rows):
    for j in range(cols):
        for d in range(deeps):
            reverse_c[i][j][d]=255-reverse_c[i][j][d]

cv2.imwrite("tangsan_img_reverse.jpg",cv2.hconcat([img,reverse_c]))
os.system("open tangsan_img_reverse.jpg")


# 直方图
import numpy as np
hist = np.array((256))
import pdb
pdb.set_trace()
for i in range(rows):
    for j in range(cols):
         tmp = gray[i][j]
         hist[tmp]=hist[tmp]+1

print(hist)



