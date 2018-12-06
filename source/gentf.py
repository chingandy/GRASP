"""Converts image data to TFRecords file format with Example protos.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from datetime import datetime
from skimage.io import imread
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2

import os
import random
from random import shuffle
import sys
import threading

import time
import os.path

import numpy as np
from shutil import copyfile
import tensorflow as tf

import autolab_core.utils as utils
from autolab_core.constants import *
from autolab_core import TensorDataset, YamlConfig
from image_augmentation import augment_img


# sys.path.append('/usr/local/lib/python2.7/site-packages')

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float64_feature(value):
  """Wrapper for inserting float64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def convert_to_example(filename,f1,f2, image_buffer64, trainlable):
  """Build an Example proto for an example.

  """
  channels = 2

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/encoded64': _bytes_feature(tf.compat.as_bytes(image_buffer64.tostring())),
      'image/channels': _int64_feature(channels),
      'image/trainlable': _float64_feature(trainlable),
      'image/name': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
      'image/f1': _bytes_feature(tf.compat.as_bytes(os.path.basename(f1))),
      'image/f2': _bytes_feature(tf.compat.as_bytes(os.path.basename(f2))),}))
  return example


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  print(serialized_example.shape)
  # Decode the record read by the reader
  features = tf.parse_single_example(serialized_example, features = {
        "image/encoded64": tf.FixedLenFeature([], tf.string),
        "image/channels": tf.FixedLenFeature([], tf.int64),
        "image/trainlable": tf.FixedLenFeature([], tf.float32),
        "image/name": tf.FixedLenFeature([], tf.string),
        "image/f1": tf.FixedLenFeature([], tf.string),
        "image/f2": tf.FixedLenFeature([], tf.string),

  })

  trainlabel = tf.cast(features["image/trainlable"], tf.float32)

  image_encoded64 = features["image/encoded64"]
  image_raw64= tf.decode_raw(image_encoded64, tf.uint8)
  image64 = tf.reshape(image_raw64,[64,64,2])

  filename = tf.cast(features["image/name"], tf.string)
  f1 = tf.cast(features["image/f1"], tf.string)
  f2 = tf.cast(features["image/f2"], tf.string)

  return image64, trainlabel, filename,f1,f2



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

def main():
  print("Start")
  # DATASETNAME= datasetfile.split(".")[-2]
  # writer = tf.python_io.TFRecordWriter(DATASETNAME+'.tfrecords')
  # DATA_DIR="./tf-dataset/"
  DATASETNAME = filepath.split("/")[-1].split(".")[-2]
  writer = tf.python_io.TFRecordWriter(DATA_DIR+DATASETNAME+'.tfrecords')
  # datasetfile = "dataset_100_200.txt"

  # DATA_DIR = "/Users/chingandywu/chinganwu/KTH/Y2P1/project-course-in-data-science/data_gen/"
  # datasetfile = "dataset_100_200.txt"


  #loop over all the lines
  counter = 0
  ccounter=0
  hist=[]
  cat_all=[]
  master_data_cage=[]
  master_data_nc=[]

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
        if trainlabel==0:
          master_data_cage.append([namech1,namech2,trainlabel,obj_c,obs_c])
        if trainlabel==1:
          master_data_nc.append([namech1,namech2,trainlabel,obj_c,obs_c])

  else:
    print(" could not find: " + datasetfile)



  #What do we got:
  print( "  # data set sizo cage-relevant: " + str(len(master_data_cage)) + " and cage-irrelevant: " + str(len(master_data_nc)) )
  shuffle(master_data_cage)
  shuffle(master_data_nc)
  master_uni=[]

  #fill master uin with caged

  for j in range(len(master_data_cage)):
    namech1,namech2,trainlable,obj_c,obs_c=master_data_cage[j]
    master_uni.append([namech1,namech2,trainlable,obj_c,obs_c])

  for i in range(len(master_data_nc)):
    namech1,namech2,trainlable,obj_c,obs_c=master_data_nc[i]
    master_uni.append([namech1,namech2,trainlable,obj_c,obs_c])


  #save trainset
  print("************************Convert txt into TFrecord ********************")
  dataset_size = len(master_data_nc) + len(master_data_cage)
  print("Original size: ", dataset_size)
  print("Cage irrelevant: ", len(master_data_nc))
  print("Cage relevant: ", len(master_data_cage))

  dataset_size = len(master_data_nc) + len(master_data_cage) * 72
  print("\nSize after augmentation : ", dataset_size)
  print("Cage irrelevant: ", len(master_data_nc))
  print("Cage relevant: ", len(master_data_cage)*72)
  if len(master_data_nc) != len(master_data_cage)*72:
      print("The data set is still uneven! But we will see.\n")



  random.shuffle(master_uni)
  # name="nothing"
  # img = np.ones((64,64), np.uint8)*255
  # cv2.circle(img,(32,32), 8, (0,0,0), -1)
  #
  # for i in range(len(master_data_nc)):
  #
  #   #make image
  #   namech1,namech2,trainlable,obj_c,obs_c = master_data_nc[i]
  #
  #   img = np.ones((64,64), np.uint8)*255
  #   img = draw_object(img,obj_c)
  #   img2 = np.ones((64,64), np.uint8)*255
  #   img2= draw_obstacles(img2,obs_c)
  #
  #   # image_data64 = np.stack((img,)*2, -1)
  #   # image_data64[:,:,1]=img2
  #   image_data64 = np.stack((img, img2), -1)
  #   name=namech2
  #   tf_example = convert_to_example(name,namech1,namech2,image_data64,trainlable)
  #   writer.write(tf_example.SerializeToString())
  #   counter += 1
  #
  #   if counter % 10000 == 0:
  #       print("Percent done", (counter/dataset_size*100))
  #
  # for i in range(len(master_data_cage)):
  #
  #   #make image
  #   namech1,namech2,trainlable,obj_c,obs_c = master_data_cage[i]
  #
  #   # the options of each operations
  #   translation_ops = [0,1,2,3,4,5,6,7,8]
  #   flip_ops = [0,1]
  #   rotation_ops = [0,1,2,3]
  #
  #   for t in translation_ops:
  #     for f in flip_ops:
  #         for r in rotation_ops:
  #
  #             img = np.ones((64,64), np.uint8)*255
  #             img = draw_object(img,obj_c)
  #             img2 = np.ones((64,64), np.uint8)*255
  #             img2= draw_obstacles(img2,obs_c)
  #             img = augment_img(img, t, f, r)
  #             img2 = augment_img(img2, t, f, r)
  #
  #             image_data64 = np.stack((img, img2), -1)
  #             name=namech2
  #             tf_example = convert_to_example(name,namech1,namech2,image_data64,trainlable)
  #             writer.write(tf_example.SerializeToString())
  #             counter += 1
  #
  #   if counter % 10000 == 0:
  #       print("Percent done", (counter/dataset_size*100))
  examples = []
  for i in range(len(master_uni)):

      #make image
      namech1,namech2,trainlable,obj_c,obs_c = master_uni[i]

      if trainlable == 0:
           # the options of each operations
           translation_ops = [0,1,2,3,4,5,6,7,8]
           flip_ops = [0,1]
           rotation_ops = [0,1,2,3]

           for t in translation_ops:
             for f in flip_ops:
                 for r in rotation_ops:

                     img = np.ones((64,64), np.uint8)*255
                     img = draw_object(img,obj_c)
                     img2 = np.ones((64,64), np.uint8)*255
                     img2= draw_obstacles(img2,obs_c)
                     img = augment_img(img, t, f, r)
                     img2 = augment_img(img2, t, f, r)

                     image_data64 = np.stack((img, img2), -1)
                     name=namech2
                     examples.append([name,namech1,namech2,image_data64,trainlable])
                     # tf_example = convert_to_example(name,namech1,namech2,image_data64,trainlable)
                     # writer.write(tf_example.SerializeToString())
                     # counter += 1
                     # if counter % 10000 == 0:
                     #     print("Percent done", (counter/dataset_size*100))

      elif trainlable == 1:
          img = np.ones((64,64), np.uint8)*255
          img = draw_object(img,obj_c)
          img2 = np.ones((64,64), np.uint8)*255
          img2= draw_obstacles(img2,obs_c)

          # image_data64 = np.stack((img,)*2, -1)
          # image_data64[:,:,1]=img2
          image_data64 = np.stack((img, img2), -1)
          name=namech2
          examples.append([name,namech1,namech2,image_data64,trainlable])
          # tf_example = convert_to_example(name,namech1,namech2,image_data64,trainlable)
          # writer.write(tf_example.SerializeToString())
          # counter += 1
          #
          # if counter % 10000 == 0:
          #     print("Percent done", (counter/dataset_size*100))

      else:
          print("SOMETHING WENT WRONG.")

  shuffle(examples)
  for name,namech1,namech2,image_data64,trainlable in examples:

      tf_example = convert_to_example(name,namech1,namech2,image_data64,trainlable)
      writer.write(tf_example.SerializeToString())
      counter += 1

      if counter % 10000 == 0:
          print("Percent done", (counter/dataset_size*100))





  # for i in range(len(master_uni)):
  #
  #   #make image
  #   namech1,namech2,trainlable,obj_c,obs_c = master_uni[i]
  #
  #   img = np.ones((64,64), np.uint8)*255
  #   img = draw_object(img,obj_c)
  #   img2 = np.ones((64,64), np.uint8)*255
  #   img2= draw_obstacles(img2,obs_c)
  #
  #   # image_data64 = np.stack((img,)*2, -1)
  #   # image_data64[:,:,1]=img2
  #   image_data64 = np.stack((img, img2), -1)
  #   name=namech2
  #   tf_example = convert_to_example(name,namech1,namech2,image_data64,trainlable)
  #   writer.write(tf_example.SerializeToString())
  #   counter += 1
  #
  #   if counter % 10000 == 0:
  #     print("Percent done", (counter/len(master_uni)*100))



  print("saved: " + str(counter))
  writer.close()
  time.sleep(1)


if __name__ == '__main__':
    print(str(sys.argv))

    if(len(sys.argv) < 2):
        print('usage: gentf.py <.txt file>, where <.txt file> is the file you want to convert to tfrecord')
        quit()

    # datasetfile= str(sys.argv[1])
    filepath = str(sys.argv[1])

    if len(sys.argv) >2:
        DATA_DIR= sys.argv[2]
    else:
        DATA_DIR="../tf-dataset/"
    main()
