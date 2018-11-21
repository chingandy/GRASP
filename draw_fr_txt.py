import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import random
from random import shuffle
from math import sin, cos, pi
from image_augmentation import augment_img

# def draw_object(img,obj_c, deg=0):
#     items=obj_c
#     count = 0
#     (x,y) = (0,0)
#     for i in range(1,(int(items[0]))*3,3):
#         x += int(items[i])
#         y += int(items[i+1])
#         count += 1
#         # cv2.circle(img,(int(items[i]),int(items[i+1])), int(items[i+2]), (0,0,0), -1)
#     center = (x/count,y/count)
#
#     for i in range(1,(int(items[0]))*3,3):
#       # print("Obstacles: ", items[i],items[i+1])
#       deg = deg * pi / 180.0
#       (x,y) = (int(items[i]), int(items[i+1]))
#       print("#"*50)
#       print("x, y = ", x, y)
#       # (cx, cy) = center
#       (cx,cy) = (31.5,31.5)
#       print("cx,cy = ", cx, cy)
#       print("sin(deg): ", sin(deg))
#       print("cos(deg): ", cos(deg))
#       items[i] = (x-cx)*cos(deg) - (y-cy)*sin(deg)
#       items[i] += cx
#       items[i+1] = (x-cx)*sin(deg) + (y-cy)*cos(deg)
#       items[i+1] += cy
#       cv2.circle(img,(int(items[i]),int(items[i+1])), int(items[i+2]), (0,0,0), -1)
#     return img, center
def draw_object(img,obj_c):
  items=obj_c
  for i in range(1,(int(items[0]))*3,3):
      cv2.circle(img,(int(items[i]),int(items[i+1])), int(items[i+2]), (0,0,0), -1)
  return img

def draw_obstacles(img,obs_c):
    items=obs_c.split(",")[1:-1]
    for i in range(1,(int(items[0]))*3,3):
            cv2.circle(img,(int(items[i]),int(items[i+1])), int(items[i+2]), (0,0,0), -1)
    return img

def draw_object_t(img,obj_c):
  items=obj_c
  for i in range(1,(int(items[0]))*3,3):
      cv2.circle(img,(int(items[i]),int(items[i+1])), int(items[i+2]), (1,2,1), -1)
  return img

def draw_obstacles_t(img,obs_c):
    items=obs_c.split(",")[1:-1]
    for i in range(1,(int(items[0]))*3,3):
            cv2.circle(img,(int(items[i]),int(items[i+1])), int(items[i+2]), (1,2,1), -1)
    return img

def rotate_img(img, angle):

    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    dst[0,:] = 255
    dst[:,0] = 255

    return dst
# def draw_obstacles_2(img,obs_c, center, deg=0.0):
#   items=obs_c.split(",")[1:-1]
#   for i in range(1,(int(items[0]))*3,3):
#     # print("Obstacles: ", items[i],items[i+1])
#     deg = deg * pi / 180.0
#     (x,y) = (int(items[i]), int(items[i+1]))
#     print("#"*50)
#     print("x, y = ", x, y)
#     # (cx, cy) = center
#     (cx,cy) = (31.5,31.5)
#     print("cx,cy = ", cx, cy)
#     print("sin(deg): ", sin(deg))
#     print("cos(deg): ", cos(deg))
#     items[i] = (x-cx)*cos(deg) - (y-cy)*sin(deg)
#     items[i] += cx
#     items[i+1] = (x-cx)*sin(deg) + (y-cy)*cos(deg)
#     items[i+1] += cy
#     # print("items[i]: ", type(items[i]))
#     print("Obstacles: ", items[i],items[i+1])
#     cv2.circle(img,(int(items[i]),int(items[i+1])), int(items[i+2]), (0,0,0), -1)
#   return img
def add_salt_pepper_noise(img):
    # Need to produce a copy as to not modify the original image
    X_img = np.copy(img)
    row, col= X_img.shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * (row*col) * salt_vs_pepper)
    num_pepper = np.ceil(amount *(row*col) * (1.0 - salt_vs_pepper))


    # Add Salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
    X_img[coords[0], coords[1]] = 1

    # Add Pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
    X_img[coords[0], coords[1]] = 0

    return X_img

def augment_data(img,n,r):
    if n == 0:
        pass
    elif n == 1:
        img = add_salt_pepper_noise(img)
    if r == 0 :
        pass
    elif r == 1:
        img = rotate_img(img, angle=90)

    return img


print("Start")
  # DATASETNAME="Dataset_all_test"
  # DATA_DIR="./"
  # datasetfile = "Dataset_test.txt"
filepath = "/Users/chingandywu/GRASP/rebuilt-dataset/test_small_dataset_300_400.txt"
 #loop over all the lines
counter = 0
ccounter=0
hist=[]
cat_all=[]
master_data_cage=[]
master_data_nc=[]
master_uni = []


if(os.path.isfile(filepath)):
  with open(filepath) as f:
    lines = f.readlines()

    for line in lines:
      items = line.split(",")
      num_c_obj=int(items[1])
      obj_c=items[1:num_c_obj*3+2]
      obs_c_str=line.split(".jpg")[2]
      obs_c=obs_c_str.split(",-,")[0]
      objstring = line.split(",-,")
      items=objstring[1].split(",")
      items=items[0:]
      namech1=items[0]
      namech2=items[1]

      trainlabel=int(items[4])
      master_uni.append([namech1,namech2,trainlabel,obj_c,obs_c])



#save trainset
print("************************Building Training set ********************")
print("size: " + str(len(master_uni)))
# random.shuffle(master_uni)
name="nothing"
img = np.ones((64,64), np.uint8)*255
cv2.circle(img,(32,32), 8, (0,0,0), -1)
plt.imshow(img)
plt.show()
for i in range(len(master_uni)):

  #make image
  namech1,namech2,trainlable,obj_c,obs_c = master_uni[i]

  img = np.ones((64,64), np.uint8)*255
  img= draw_object(img,obj_c)
  img= draw_obstacles(img,obs_c)
  # img2 = rotate_img(img, 90)
  # img3 = add_salt_pepper_noise(img)
  img2 = augment_img(img, 1,0,0)
  img3 = augment_img(img, 1, 1,0)
  img4 = augment_img(img,1,1,1)

  plt.subplot(141)
  plt.imshow(img)
  plt.title("Original")

  plt.subplot(142)
  plt.imshow(img2)
  plt.title("Rotate 90")

  plt.subplot(143)
  plt.imshow(img3)
  plt.title("Noise")

  plt.subplot(144)
  plt.imshow(img4)
  plt.title("Noise")
  plt.show()
