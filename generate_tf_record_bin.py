
"""Converts image data to TFRecords file format with Example protos.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from datetime import datetime
from skimage.io import imread
import matplotlib
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

#import autolab_core.utils as utils
#from autolab_core.constants import *
#from autolab_core import TensorDataset, YamlConfig

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
  print(trainlabel.shape)

  image_encoded64 = features["image/encoded64"]
  image_raw64= tf.decode_raw(image_encoded64, tf.uint8)
  image64 = tf.reshape(image_raw64,[64,64,2])

  filename = tf.cast(features["image/name"], tf.string)
  f1 = tf.cast(features["image/f1"], tf.string)
  f2 = tf.cast(features["image/f2"], tf.string)

  return image64, trainlabel, filename,f1,f2

def test_tf_record(DATASETNAME):
  with tf.Session() as sess:
    print("decoding tf file")    
    filenames = [os.path.join(DATASETNAME +'.tfrecords')]
    for f in filenames:
          if not tf.gfile.Exists(f):
              raise ValueError("Failed to find file: " + f)
    filename_queue = tf.train.string_input_producer(filenames)

    image64, trainlabel, filename,f1,f2 = read_and_decode(filename_queue)

    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(5):
      reconstructed_img64,re_filename,re_f1,re_f2,re_trainlabel = sess.run([image64,filename,f1,f2,trainlabel])
       
      print("reconstructed: " + str(re_filename) + " ,f1: " + str(re_f1) + " ,f2: " + str(re_f2) + " , trainlable: " + str(re_trainlabel) )

      plt.subplot(1, 2, 1)
      plt.imshow(reconstructed_img64[:,:,0])
      plt.title('Object64')
      plt.subplot(1, 2, 2)
      plt.imshow(reconstructed_img64[:,:,1])
      plt.title('grippers64')

      plt.show()

    coord.request_stop()
    coord.join(threads)


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
  DATASETNAME="Dataset_small2"
  writer = tf.python_io.TFRecordWriter(DATASETNAME+'.tfrecords')
  DATA_DIR="./"
  datasetfile="Dataset_small2.txt"
  #loop over all the lines
  counter = 0
  ccounter=0
  hist=[]
  cat_all=[]
  master_data_cage=[]
  master_data_nc=[]
  if(os.path.isfile(DATA_DIR + datasetfile)):
    with open(DATA_DIR + datasetfile) as f:
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
          
          #Rotate the image 180degrees and and append it.
          print("namech1 and namech2 ", namech1, namech2)
          print("This is obj ",obj_c)
          print("This is obs ",obs_c)
          namech1_rot = namech1[:-4] + 'rot180' + namech1[-4:]
          namech2_rot = namech2[:-4] + 'rot180' + namech1[-4:]
          print(namech1_rot)
          print(namech2_rot)
          newobj = []
          newobs = []
          
          #Rotate the object
          for i in range(len(obj_c)):
              if i%3 != 0:
                  k = abs(int(obj_c[i])-64)
                  newobj.append(str(k))
              else:
                  newobj.append(obj_c[i])
          print("Done with object")
          print("This is what it looks like ", newobj)
          tmp = obs_c.split(",")[1:-1]
          #print(tmp)
          #Rotate the grippers
          for i in range(len(tmp)):
              if i%3 != 0:
                  k = abs(int(tmp[i])-64)
                  newobs.append(str(k))
              else:
                  newobs.append(tmp[i])
                  
          newobs.append(" ")
          newobs.insert(0, " ")
          newobs = ','.join(newobs)
          print(newobs)
          master_data_cage.append([namech1_rot,namech2_rot,trainlabel,newobj,newobs])

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
  print("************************Building Taining set ********************")
  print("size: " + str(len(master_uni)))
  random.shuffle(master_uni)
  name="nothing"
  img = np.ones((64,64), np.uint8)*255
  cv2.circle(img,(32,32), 8, (0,0,0), -1)
  for i in range(len(master_uni)):
    
    #make image
    namech1,namech2,trainlable,obj_c,obs_c = master_uni[i]
    
    img = np.ones((64,64), np.uint8)*255
    img = draw_object(img,obj_c)
    img2 = np.ones((64,64), np.uint8)*255
    img2= draw_obstacles(img2,obs_c)
  
    image_data64 = np.stack((img,)*2, -1)
    image_data64[:,:,1]=img2
    name=namech2
    tf_example = convert_to_example(name,namech1,namech2,image_data64,trainlable)
    writer.write(tf_example.SerializeToString())
    counter += 1

    if counter % 10000 == 0:
      print("Percent done", (counter/len(master_uni)*100))
    

  print("saved: " + str(counter))
  writer.close()
  time.sleep(1)
  test_tf_record(DATASETNAME)


if __name__ == '__main__':
  main()
