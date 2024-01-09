#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import cv2


# In[3]:


# lm = pd.read_json('C:/Users/ankit/OneDrive - iitkgp.ac.in/Desktop/ISP/Thermal Face Detection/thermal-facial-landmarks-detection/dataset/iron/train/json/100_1_2_1_1134_36_1.json')


# In[4]:


data = json.load(open('C:/Users/ankit/OneDrive - iitkgp.ac.in/Desktop/ISP/Thermal Face Detection/thermal-facial-landmarks-detection/dataset/iron/train/json/100_1_2_1_1134_36_1.json'))

lm = pd.DataFrame(data["shapes"])


# In[5]:


# lm


# In[6]:


nose_bridge = np.array((lm[lm['label']=='nose_bridge']['points']).item(),dtype=float)


# In[7]:


nose_tip = np.array((lm[lm['label']=='nose_tip']['points']).item(),dtype=float)


# In[8]:


# nose_tip


# In[9]:


#4th point of left eye
left_eye = np.array((lm[lm['label']=='left_eye']['points']).item(),dtype=float)


# In[10]:


#1st point of right eye
right_eye = np.array((lm[lm['label']=='right_eye']['points']).item(),dtype=float)


# In[11]:


t1 = 0.1
nose_upper_left = np.array([nose_tip[0][0]+0.02*nose_tip[0][0],left_eye[4][1]],dtype=int)


# In[12]:


t2 = 0.9
nose_lower_right = np.array([nose_tip[4][0]*t2+(1-t2)*right_eye[0][0], nose_tip[0][1]],dtype=int)


# In[13]:


# nose_lower_right


# In[14]:


# nose_upper_left


# In[15]:


img = cv2.imread('C:/Users/ankit/OneDrive - iitkgp.ac.in/Desktop/ISP/Thermal Face Detection/thermal-facial-landmarks-detection/dataset/iron/train/images/100_1_2_1_1134_36_1.png')


# In[16]:


# cv2.imshow('win',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[17]:


cv2.rectangle(img, nose_upper_left,nose_lower_right, (0,255,0), 2)

# cv2.namedWindow("image", cv2.WINDOW_NORMAL)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[18]:


chin = np.array((lm[lm['label']=='chin']['points']).item(),dtype=float)


# In[19]:


# chin


# In[20]:


# left_chin_upper = np.array([chin[4][0],0.5*chin[1][1]+0.5*chin[2][1]], dtype=int)
left_chin_upper = np.array([chin[4][0],chin[1][1]], dtype=int)


# In[21]:


lip_p1 = np.array((lm[lm['label']=='lip_p1']['points']).item(),dtype=float)


# In[22]:


left_chin_lower = np.array([chin[6][0], chin[4][1]], dtype=int)


# In[23]:


cv2.rectangle(img, left_chin_upper,left_chin_lower, (0,255,0), 2)

# cv2.namedWindow("image")
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[24]:


right_chin_upper = np.array([right_eye[5][0],chin[15][1]], dtype=int)
right_chin_lower = np.array([chin[12][0], chin[12][1]], dtype=int)


# In[25]:


left_eyebrow = np.array((lm[lm['label']=='left_eyebrow']['points']).item(),dtype=float)


# In[26]:


# left_eyebrow


# In[27]:


right_eyebrow = np.array((lm[lm['label']=='right_eyebrow']['points']).item(),dtype=float)


# In[28]:


# right_eyebrow


# In[29]:


forehead_top_left = max(left_eyebrow[3][1],right_eyebrow[1][1])-0.6*(abs(nose_lower_right[1]-nose_upper_left[1]))


# In[30]:


forehead_top_left = np.array([left_eyebrow[2][0],forehead_top_left], dtype=int)


# In[31]:


forehead_bottom_right = np.array([right_eyebrow[2][0],min(left_eyebrow[3][1],right_eyebrow[2][1])-0.1], dtype=int)


# In[32]:


# right_eyebrow[2][0]


# In[33]:


# forehead_bottom_right


# In[34]:


cv2.rectangle(img,forehead_top_left, forehead_bottom_right, (0,255,0),2)


# In[35]:


cv2.rectangle(img, right_chin_upper,right_chin_lower, (0,255,0), 2)

cv2.namedWindow("image")
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




