import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import random
from random import shuffle
from math import sin, cos, pi


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

#Rotate image by angle
def rotate_img(img, angle):
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    dst[0,:] = 255
    dst[:,0] = 255
    return dst

#Flip image vertically
def flip_img(img):

    vertical_img = cv2.flip(img, 1)
    return vertical_img

#Move picture one pixel to the right
def translate_right(img):
    rows,cols = img.shape
    M = np.float32([[1,0,1],[0,1,0]])
    translated_img = cv2.warpAffine(img,M,(cols,rows))
    translated_img[:,0] = 255
    return translated_img

#Move picture one pixel to the left
def translate_left(img):
    rows,cols = img.shape
    M = np.float32([[1,0,-1],[0,1,0]])
    translated_img = cv2.warpAffine(img,M,(cols,rows))
    translated_img[:,-1] = 255
    return translated_img

#Move picture one pixel up
def translate_up(img):
    rows,cols = img.shape
    M = np.float32([[1,0,0],[0,1,-1]])
    translated_img = cv2.warpAffine(img,M,(cols,rows))
    translated_img[-1,:] = 255
    return translated_img

#Move picture one pixel down 
def translate_down(img):
    rows,cols = img.shape
    M = np.float32([[1,0,0],[0,1,1]])
    translated_img = cv2.warpAffine(img,M,(cols,rows))
    translated_img[0,:] = 255
    return translated_img

def augment_img(img, rot, flip, translate):
    #Do translations on the image
    if translate == 0:
        aug_img = img
    elif translate == 1:
        aug_img = translate_right(img)
    elif translate == 2:
        aug_img = translate_left(img)
    elif translate == 3:
        aug_img = translate_up(img)
    elif translate == 4:
        aug_img = translate_down(img)
    else: 
        print("Not valid argument for translation")

    #Flip the image
    if flip == 0:
        pass
    elif flip == 1:
        aug_img = flip_img(aug_img)
    else:
        print("Not valid argument for flipping")

    #Rotate the image
    if rot == 0:
        pass
    elif rot == 1:
        aug_img = rotate_img(aug_img, 90)
    elif rot = 2:
        aug_img = rotate_img(aug_img, 180)
    elif rot = 3:
        aug_img = rotate_img(aug_img, 270)
    else:
        print("Not valid argument for rotation")

    return aug_img

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


print("Start")
  # DATASETNAME="Dataset_all_test"
  # DATA_DIR="./"
  # datasetfile = "Dataset_test.txt"
filepath = "Dataset_small2.txt"
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
      # if trainlabel==0:
      #   master_data_cage.append([namech1,namech2,trainlabel,obj_c,obs_c])
      # if trainlabel==1:
      #   master_data_nc.append([namech1,namech2,trainlabel,obj_c,obs_c])
      master_uni.append([namech1,namech2,trainlabel,obj_c,obs_c])



#What do we got:
# print( "  # data set sizo cage-relevant: " + str(len(master_data_cage)) + " and cage-irrelevant: " + str(len(master_data_nc)) )
# shuffle(master_data_cage)


#fill master uin with caged
#
# for j in range(len(master_data_cage)):
#   namech1,namech2,trainlable,obj_c,obs_c=master_data_cage[j]
#   master_uni.append([namech1,namech2,trainlable,obj_c,obs_c])
#
# for i in range(len(master_data_nc)):
#   namech1,namech2,trainlable,obj_c,obs_c=master_data_nc[i]
#   master_uni.append([namech1,namech2,trainlable,obj_c,obs_c])


#save trainset
print("************************Building Training set ********************")
print("size: " + str(len(master_uni)))
# random.shuffle(master_uni)
name="nothing"
img = np.ones((64,64), np.uint8)*255
cv2.circle(img,(32,32), 8, (0,0,0), -1)
#plt.imshow(img)
#plt.show()
for i in range(len(master_uni)):

  #make image
  namech1,namech2,trainlable,obj_c,obs_c = master_uni[i]

  img = np.ones((64,64), np.uint8)*255

  print("img: ", img.dtype)
  img= draw_object(img,obj_c)
  # img2 = np.ones((64,64), np.uint8)*255
  img= draw_obstacles(img,obs_c)
  vertical_img = flip_img(img)
  # img2 = np.ones((64,64), np.uint8)*255
  # img2, center = draw_object(img2,obj_c, deg=180)
  # # img2 = np.ones((64,64), np.uint8)*255
  # img2= draw_obstacles(img2,obs_c, center, deg=180)

  img2 = rotate_img(img, 90)
  img3 = rotate_img(img, 180)
  img4 = rotate_img(img, 270)

  # img3 = np.ones((64,64), np.uint8)*255
  # img3 = draw_object(img,obj_c)
  # img3 = draw_obstacles(img,obs_c)
  # plt.imshow(img)
  # plt.subplot(1, 4, 1)
  # plt.imshow(img)
  # plt.subplot(1, 4, 2)
  # plt.imshow(img2)
  # plt.subplot(1,4,3)
  # plt.imshow(img3)
  # plt.subplot(1,4,4)
  img5 = np.arange(64*64,dtype=np.uint8).reshape(64,64)

  # img3 = draw_object(img3, obj_c)
  print("img3: ", img3.dtype)
  img5 = draw_object(img5, obj_c)
  img5 = draw_obstacles(img5, obs_c)
  img6 = translate_down(img)
  plt.subplot(331)
  plt.imshow(img)
  plt.subplot(332)
  plt.imshow(img2)
  plt.subplot(333)
  plt.imshow(img3)
  plt.subplot(334)
  plt.imshow(img4)
  plt.subplot(335)
  plt.imshow(img5)
  plt.subplot(336)
  plt.imshow(vertical_img)
  plt.subplot(337)
  plt.imshow(img6)
  plt.show()
