import cv2
import torch
def show_featuremap(feature):
    b,c,w,h = feature.shape
    feature_show=torch.zeros(b*w,c*h)
    for bi in range(0,b):
        for ci in range(0,c):
            #import pdb
            #pdb.set_trace()
            feature_show[bi*w:(bi+1)*w,ci*h:(ci+1)*h]=feature[bi,ci,:,:]
    print(feature_show.shape)
    cv2.imwrite("feature_bw_ch.png",255*feature_show.float().numpy())
    import os
    os.system("open feature_bw_ch.png")
    return feature_show

if __name__=="__main__":
    feature = torch.randn(1,10,100,100)
    feature_show = show_featuremap(feature)
    print(feature_show.shape)
    cv2.imwrite("feature_bw_ch.png",255*feature_show.float().numpy())
    import os
    os.system("open feature_bw_ch.png")
