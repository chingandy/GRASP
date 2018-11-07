import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

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
  return image64, trainlabel 

def read_dataset(DATASETNAME):
    batch_size = 7
    data = []
    i = 1
    with tf.Session() as sess:
        print("decoding tf file")
        # filenames = [os.path.join(DATASETNAME +'.tfrecords')]

        """ test how to make batch input data"""
        filenames = tf.placeholder(tf.string, shape=[None])
        dataset = tf.data.TFRecordDataset(filenames)

        # Map the parser over dataset, and batch results by up to batch_size
        dataset = dataset.map(parser,num_parallel_calls=None)
        #dataset = dataset.batch(batch_size)
        #dataset = dataset.repeat()
        iterator = dataset.make_initializable_iterator()
        training_filenames = [os.path.join(DATASETNAME +'.tfrecords')]
        sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

        feature, label = iterator.get_next()
        features = []
        labels = []
        try:
            while True:
                f, l = sess.run([feature, label])
                features.append(f)
                labels.append(l)
                print("Has looped: ", i)
                i+=1
        except tf.errors.OutOfRangeError:
            pass
        npfeatures = np.array(features, dtype=np.int32)
        nplabels = np.array(labels, dtype=np.int32)
        print("Returning features and labels")
    return npfeatures, nplabels

def main():
    DATASETNAME = "../../Data_science_dataset/Dataset_small_test"
    # test_tf_record(DATASETNAME)
    reconstructed_img64, re_filename, re_f1, re_f2, re_trainlabel=read_dataset(DATASETNAME)
    print("Printing in Main function:")
    print("reconstructed_img64 TYPE: ", type(reconstructed_img64))
    print("reconstructed_img64 SIZE: ", reconstructed_img64.shape)
    print("re_trainlabel TYPE:", type(re_trainlabel))
    print("re_trainlabel SIZE:", re_trainlabel.shape)


if __name__ == '__main__':
    main()
