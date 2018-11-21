import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

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


def parser(serialized_example):
  """Parses a single tf.Example into image and label tensors."""
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

def read_dataset(DATASETNAME):
    batch_size = 5
    with tf.Session() as sess:
      print("decoding tf file")
      # filenames = [os.path.join(DATASETNAME +'.tfrecords')]

      """ test how to make batch input data"""
      filenames = tf.placeholder(tf.string, shape=[None])
      dataset = tf.data.TFRecordDataset(filenames)

      # Map the parser over dataset, and batch results by up to batch_size
      dataset = dataset.map(parser,num_parallel_calls=None)
      dataset = dataset.batch(batch_size)
      dataset = dataset.repeat()
      # print("#"*50)
      # print("DATASET: ", sess.run(dataset))
      # iterator = dataset.make_one_shot_iterator()
      iterator = dataset.make_initializable_iterator()
      training_filenames = [os.path.join(DATASETNAME +'.tfrecords')]
      sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

      image64, trainlabel, filename,f1,f2 = iterator.get_next()

      reconstructed_img64,re_filename,re_f1,re_f2,re_trainlabel = sess.run([image64,filename,f1,f2,trainlabel])


      return reconstructed_img64, re_filename, re_f1, re_f2, re_trainlabel

def read_dataset_2(DATASETNAME, size):
    image_set = []
    label_set = []
    count = 0
    with tf.Session() as sess:
      print("Reading tf file:")
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

      for i in range(size):
          reconstructed_img64,_,_,_,re_trainlabel = sess.run([image64,filename,f1,f2,trainlabel])
          image_set.append(reconstructed_img64)
          label_set.append(re_trainlabel)
          if count % 1000 == 0:
              print("Reading data: ", count)
          count += 1

      coord.request_stop()
      coord.join(threads)


      image_set = np.array(image_set)
      label_set = np.array(label_set)
      return image_set, label_set

def read_dataset_(DATASETNAME):
    image_set = []
    label_set = []
    count = 0
    with tf.Session() as sess:
      print("Reading tf file:")
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

      try:
          while True:
              reconstructed_img64,_,_,_,re_trainlabel = sess.run([image64,filename,f1,f2,trainlabel])
              image_set.append(reconstructed_img64)
              label_set.append(re_trainlabel)
              if count % 1000 == 0:
                  print("Counter: ", count)
              count += 1
      except tf.errors.OutOfRangeError as e:
          coord.request_stop(e)

      finally:
          coord.request_stop()
          coord.join(threads)


      image_set = np.array(image_set)
      label_set = np.array(label_set)
      return image_set, label_set

""" data augmentation """

def rotate_img(img, angle):

    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    dst[0,:] = 255
    dst[:,0] = 255

    return dst

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

#Move picture one pixel to the right
def translate_right(img):
    rows,cols = img.shape
    M = np.float32([[1,0,3],[0,1,0]])
    translated_img = cv2.warpAffine(img,M,(cols,rows))
    translated_img[:,0:3] = 255
    return translated_img


#Move picture one pixel to the left
def translate_left(img):
    rows,cols = img.shape
    M = np.float32([[1,0,-3],[0,1,0]])
    translated_img = cv2.warpAffine(img,M,(cols,rows))
    translated_img[:,61:64] = 255
    return translated_img

#Move picture one pixel up
def translate_up(img):
    rows,cols = img.shape
    M = np.float32([[1,0,0],[0,1,-3]])
    translated_img = cv2.warpAffine(img,M,(cols,rows))
    translated_img[61:64,:] = 255
    return translated_img

#Move picture one pixel down
def translate_down(img):
    rows,cols = img.shape
    M = np.float32([[1,0,0],[0,1,3]])
    translated_img = cv2.warpAffine(img,M,(cols,rows))
    translated_img[0:3,:] = 255
    return translated_img
def main():
    # DATASETNAME = "Data_science_dataset/Dataset_all_test"
    DATASETNAME = "/Users/chingandywu/GRASP/dataset_100_200"
    # test_tf_record(DATASETNAME)
    image_1,  label_1=read_dataset_2(DATASETNAME)
    # image_2,  label_2=read_dataset_2(DATASETNAME)

    # print("filename_1: ", filename_1)
    # print("filename_2: ", filename_2)
    print("Printing in Main function:")
    print(type(image_1))
    print(image_1.shape)
    if np.all(image_1[0] == image_1[3]):
        print("the same")
    else:
        print("different")

    # print("reconstructed_img64 TYPE: ", type(reconstructed_img64))
    # print("reconstructed_img64 SIZE: ", reconstructed_img64.shape)
    # print("re_trainlabel TYPE:", type(re_trainlabel))
    # print("re_trainlabel SIZE:", re_trainlabel.shape)




if __name__ == '__main__':
    # DATASETNAME = "/Users/chingandywu/GRASP/dataset_100_200"
    # reconstructed_img64, re_filename, re_f1, re_f2, re_trainlabel = read_dataset(DATASETNAME)
    # print("reconstructed_img64.type: \n", type(reconstructed_img64))
    # print("reconstructed_img64.size: \n", reconstructed_img64.shape)

    main()
