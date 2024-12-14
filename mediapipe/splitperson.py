import cv2
import mediapipe as mp
import numpy as np
 
# 获取pose模块
mp_pose=mp.solutions.pose
# 绘图工具模块
mp_draw=mp.solutions.drawing_utils
# 获取Pose对象
pose=mp_pose.Pose(static_image_mode=True, enable_segmentation=True)
 
# 获取背景,原图
bg=cv2.imread('bg.JPG')
im=cv2.imread('img.JPG')
cv2.imshow('bg',bg)
cv2.imshow('im',im)
# 将背景的size设置和原图size一致
w,h,c=im.shape
bg=cv2.resize(bg,(h,w))
 
# 使用Pose对象处理图像，得到姿态的关键点
im_rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
result=pose.process(im_rgb)
# cv2.imshow('result',result)
 
# segmentation_mask中的数据值是0.0-1.0，值越大，表示越接近是人
mask=result.segmentation_mask
cv2.imshow('mask',mask)
 
#
# 将单通道的mask变成三通道
mask=np.stack((mask,mask,mask),-1)
# 大于0.5的才是人
mask=mask>0.5
img1=np.where(mask,im,bg)
cv2.imshow('im1',img1)
 
cv2.waitKey(0)
cv2.destroyAllWindows()