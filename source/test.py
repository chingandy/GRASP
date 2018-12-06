import tensorflow as tf
import numpy as np
import re
import random
import matplotlib.pyplot as plt
import cv2

from math import sin, cos, pi
from random import shuffle
x = np.array([1,2,3])
y = np.array([0,2,3])
if np.all(3 == x):
    print("the same")
else:
    print("Different!!")
# x = [1,2,3,4,5,6]
# y = [111,232,345,566]
# print(x)
# shuffle(x)
# print(x)
# x.append(y)
# print(x)
# shuffle(x)
# print(x)

# x = [[1,2,3,4], [4,5,6,6],[10,32,25,98]]
# shuffle(x)
# print(x)
# for i,j,k,l in x:
#     print(i,j,k,l)





#
# img = np.ones((64,64), np.uint8)*255
# cv2.circle(img,(32,32), 8, (0,0,0), -1)
# plt.imshow(img)
# plt.show()

# # x = 2
# deg = 30.0 * pi/180.0
# # print("sin: ", x*sin(deg))
# print(sin(deg))
# x = [1,2,3,4,5]
# for i in x:
#     print(i)


# str = "I am Andy Wu from TW."
#
# re = tf.compat.as_bytes(str)
# print(re)
# # with tf.Session() as sess:
# #     print(sess.run(re))
# x = tf.constant([1,2,3,4,5,6,7,8])
# result = tf.reshape(x, [-1,2,2])
# with tf.Session() as sess:
#     # sess.run(result)
#     print(sess.run(result))
# x = 10
# # x_float = float(x)
# x_float = np.float32()
#
# print("x: ", x)
# print("x_float: ", x_float)
# print(type(x_float))
# x = tf.constant([1.8, 2.3], dtype=tf.float32)
# x = tf.cast(x, tf.int32)
# with tf.Session() as sess:
#     x = tf.constant([1.8, 2.3], dtype=tf.float32)
#     print(x.eval())
#     x = tf.cast(x, tf.int32)
#     print(x.eval())

# x = np.array([1,2,3,5])
# y = np.array([1,2,3,4])
# if np.all(x == y):
#     print("same")
# import sys
# def main():
#     # filepath = "/Users/chingandywu/GRASP/data_gen/dataset_100_200.txt"
#     print(filepath.split("/")[-1].split(".")[-2])
#
#
# if __name__ == '__main__':
#     if len(sys.argv) < 2:
#         print("fail")
#         quit()
#     print(sys.argv)
#     filepath = sys.argv[1]
#     main()

#
# x = np.array([[1,2,3],[0,9, 20]])
# y = np.array([[2,3,4],[3,4,5]])
# m = []
# print("x: ", x.shape)
# m.append(x)
# m.append(y)
# print(m)
# print(type(m))
# m = np.array(m)
# print(m.shape)
# print(type(m))

# dic = {"andy": 5, "cindy": 6, "apple":10}
# print(len(dic.keys()))
# print(dic.keys()[2]) d
#
# with open("/Users/chingandywu/GRASP/rebuilt-dataset/test.txt", "r") as r:
#     line = r.readline()
#     print(line)

# x = [1,2,3,5,6,7,8,8]
# print(random.choice(x))


# dic = {"andy": [1,2,3], "cindy": [4,5,6], "apple":[7,8,9]}
# print(random.choice(dic["andy"]))

# l1 = [1,2,3,4,5]
# l2 = [1,2,3,4]
#
# if l1==l2:
#     print("the same")
# else:
#     print("different")

# def file_len(filename):
#     """ count the number of lines in the file """
#
#     with open(filename) as f:
#         for i, l in enumerate(f):
#             pass
#     return i + 1
#
# filepath_train = "/Users/chingandywu/GRASP/rebuilt-dataset/test.txt"
# filepath_test  = "/Users/chingandywu/GRASP/rebuilt-dataset/train.txt"
# train_size = file_len(filepath_train)
# test_size = file_len(filepath_test)
# print(train_size)
# print(test_size)
# from itertools import cycle
#
# lst = ['a', 'b', 'c']
#
# pool = cycle(lst)
# count = 0
# for item in pool:
#     if count < 10:
#         print (item)
#         count += 1
#     else:
#         quit()
#
# from data import parser
# training_dataset = "/Users/chingandywu/GRASP/re_dataset_100_200.tfrecords"
# batch_size = 5
# with tf.Session() as sess:
#   print("Testing........")
#
#   """ test how to make batch input data"""
#   filenames = tf.placeholder(tf.string, shape=[None])
#   dataset = tf.data.TFRecordDataset(filenames)
#
#   # Map the parser over dataset, and batch results by up to batch_size
#   dataset = dataset.map(parser,num_parallel_calls=None)
#   dataset = dataset.batch(batch_size)
#   dataset = dataset.repeat()
#   # print("#"*50)
#   # print("DATASET: ", sess.run(dataset))
#   # iterator = dataset.make_one_shot_iterator()
#   iterator = dataset.make_initializable_iterator()
#   # training_filenames = [os.path.join(DATASETNAME +'.tfrecords')]
#   sess.run(iterator.initializer, feed_dict={filenames:[training_dataset]})
#
#   image64, trainlabel, filename,f1,f2 = iterator.get_next()
#
#   img1,_,_,_,_ = sess.run([image64,filename,f1,f2,trainlabel])
#
#   img2,_,_,_,_ = sess.run([image64,filename,f1,f2,trainlabel])
#
#   img3,_,_,_,_ = sess.run([image64,filename,f1,f2,trainlabel])
#
#   # if np.all(img1==img2):
#   #     print("The same")
#   # else:
#   #     print("different!!!!")
#   # if np.all(img2 == img3):
#   #     print("The same")
#   # else:
#   #     print("different!!")
#   if np.all(sess.run([image64,filename,f1,f2,trainlabel])[0] == sess.run([image64,filename,f1,f2,trainlabel])[0]):
#       print("The same")
#   else:
#       print("different!!!!")
#
# class GQCnnDenoisingWeights(object):
#     """ Struct helper for storing weights """
#
#     def __init__(self):
#         pass
#
# x = GQCnnDenoisingWeights()
# print(x)

# import tensorflow as tf
# import numpy as np
#
# x=tf.constant([[1,2,3],[4,5,6]])
# y=[[1,2,3],[4,5,6]]
# z=np.arange(24).reshape([2,3,4])
#
# sess=tf.Session()
# # tf.shape()
# x_shape=tf.shape(x)                    #  x_shape 是一个tensor
# y_shape=tf.shape(y)                    #  <tf.Tensor 'Shape_2:0' shape=(2,) dtype=int32>
# z_shape=tf.shape(z)                    #  <tf.Tensor 'Shape_5:0' shape=(3,) dtype=int32>
# print (sess.run(x_shape))            # 结果:[2 3]
# print (sess.run(y_shape))           # 结果:[2 3]
# print (sess.run(z_shape))           # 结果:[2 3 4]
#
#
# #a.get_shape()
# x_shape=x.get_shape()  # 返回的是TensorShape([Dimension(2), Dimension(3)]),不能使用 sess.run() 因为返回的不是tensor 或string,而是元组
# print(x_shape)
# x_shape=x.get_shape().as_list()[0]
# print(x_shape) # 可以使用 as_list()得到具体的尺寸，x_shape=[2 3]
# y_shape=y.get_shape()  # AttributeError: 'list' object has no attribute 'get_shape'
# z_shape=z.get_shape()  # AttributeError: 'numpy.ndarray' object has no attribute 'get_shape'
# def make_one_hot(dataset):
#     new_dataset = []
#     for e in dataset:
#         vec = np.zeros(10)
#         vec[e] = 1;
#         new_dataset.append(vec)
#     return np.array(new_dataset)
#
# data = [1,2,4,3,7,6,9,8,0,2,3,4,6,7,2,3,0]
# data = np.reshape(data, (-1,1))
# print(data.shape)

# new_data = make_one_hot(data)
# print(new_data)
# print(new_data.shape)
# x = [[1,2,3,4],[5,6,7,8]]
# y = [[1,3],[2,4],[9,7]]
# z = [1,2,3,3]
# result = tf.matmul(y,x)
# re2 = tf.add(result,z)
# with tf.Session() as sess:
#     print(x)
#     print(y)
#     print(sess.run(result))
#     print(sess.run(re2))

# fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
# fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])

# temp1 = tf.constant([1,2,4,4])
# temp2 = tf.constant((1,2,4,5))
# sess = tf.Session()
# if sess.run(tf.reduce_all(tf.equal(temp1, temp2))):
#     print('temp1 is equal temp2')
# else:
#     print('temp1 is not equal temp2')

# iter = 100
# a = tf.Variable(0)
# iter = tf.constant(iter)
# # a = a.assign(a+iter)
# count = a + iter
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
# # a = a + iter
# with tf.Session() as sess:
#     # sess.run(init)
#
#     saver.restore(sess, "/Users/chingandywu/GRASP/checkpoint/test.ckpt")
#     print("Model restored.")
#
#     print("a: ", sess.run(a))
#     print("iter: ", sess.run(iter))
#     a = count
#     print("count: ", a)
#     save_path = saver.save(sess, "/Users/chingandywu/GRASP/checkpoint/test.ckpt")
#     print("Model saved in path: %s" % save_path)
#
# #
# tf.reset_default_graph()
# # Create some variables.
# v1 = tf.get_variable("v1", [3], initializer = tf.zeros_initializer)
# v2 = tf.get_variable("v2", [5], initializer = tf.zeros_initializer)
#
# # Add ops to save and restore only `v2` using the name "v2"
# saver = tf.train.Saver({"v2": v2})
#
# # Use the saver object normally after that.
# with tf.Session() as sess:
#   # Initialize v1 since the saver will not.
#   v1.initializer.run()
#   saver.restore(sess, "/Users/chingandywu/GRASP/checkpoint/test.ckpt")
#
#   print("v1 : %s" % v1.eval())
#   print("v2 : %s" % v2.eval())

# x = [[1,2,3,4,5,-1],[5,4,22,89,43,100]]
# x = np.array(x)
# print(x.shape)
# print(x)
# print(np.max(x))
# print(np.min(x))
# x = x/100
# print(x)
# print(np.max(x))
# print(np.min(x))
#
# x = np.array([1,0,0,0])
# print(np.nonzero(x)[0][0])
#
# for i, axis in enumerate([None, 0]):
#     print(i)
#     print(axis)

#
# save_path_1 = "/Users/chingandywu/GRASP/test/text1.txt"
# save_path_2 = "/Users/chingandywu/GRASP/test/text2.txt"
# with open(save_path_1, 'w') as a, open(save_path_2, 'w') as b:
#     a.write("just a test ")
#     b.write("another test")
# from itertools import cycle
# x = [1,2,4,6,32,76,58,21]
# pool = cycle(x)
# count = 0
# for key, item in enumerate(pool):
#     print(key, item)
#     count += 1
#     if count > 100:
# x = [1,0,9,-3,2,0,-3]
# print(np.count_nonzero(x))
