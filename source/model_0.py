from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from data import read_dataset, read_dataset_2, parser
from preprcs import separate_classes

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
  # print()
  # print("#"*80)
  # print("input_layer:", input_layer.dtype)
  # print("input_layer:", input_layer.get_shape())
  input_layer = features["x"]
  input_layer = tf.cast(input_layer, tf.float32)
  # print("#"*80)
  # print(input_layer.get_shape())
  # print(input_layer.dtype)
  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding="same")

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 16 * 16 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



def file_len(filename):
    """ count the number of lines in the file """

    with open(filename) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def main(unused_argv):

    # training_dataset = "/Users/chingandywu/GRASP/re_dataset_100_200"
    # test_dataset = "/Users/chingandywu/GRASP/re_dataset_300_400"
    # filepath_train = "/Users/chingandywu/GRASP/rebuilt-dataset/re_dataset_100_200.txt"
    # filepath_test  = "/Users/chingandywu/GRASP/rebuilt-dataset/re_dataset_300_400.txt"
    # train_size = file_len(filepath_train)
    # print("Training size: ", train_size)
    # # test_size = file_len(filepath_test)
    # test_size = 100
    # print("Test size: ", test_size)

    # training_dataset = "/Users/chingandywu/GRASP/dataset_100_200"
    # test_dataset = training_dataset
    # filepath_train = "/Users/chingandywu/GRASP/data_gen/dataset_100_200.txt"
    # filepath_test  = filepath_train
    # train_size = file_len(filepath_train)
    # test_size = train_size
    # print("SIZE: ", train_size)
    # Load training and eval data
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images # Returns np.array
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # eval_data = mnist.test.images # Returns np.array
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    """ grasp data read in """
    # train_data, train_labels=read_dataset_2(training_dataset, train_size)
    # eval_data,  eval_labels=read_dataset_2(test_dataset, test_size)
    # train_labels = np.int32(train_labels)
    # eval_labels = np.int32(eval_labels)

    """ Feed in the data in a more direct way """
    batch_size = 5
    with tf.Session() as sess:
      print("decoding tf file")

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
      # training_filenames = [os.path.join(DATASETNAME +'.tfrecords')]
      training_dataset = "re_dataset_100_200.tfrecords"
      sess.run(iterator.initializer, feed_dict={filenames:[training_dataset]})

      image64, trainlabel, filename,f1,f2 = iterator.get_next()
      img, label = sess.run([image64, trainlabel])
      print("#"*100)
      print(img.shape)
      print(label.shape)





    # # Create the Estimator
    # classifier = tf.estimator.Estimator(
    # model_fn=cnn_model_fn, model_dir="/Users/chingandywu/GRASP/model_checkpoint2")
    #
    # # Set up logging for predictions
    # tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(
    # tensors=tensors_to_log, every_n_iter=50)
    #
    # # Train the model
    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x":train_data},
    #     y=train_labels,
    #     batch_size=10,
    #     num_epochs=None,
    #     shuffle=True)
    #
    # classifier.train(
    #     input_fn=train_input_fn,
    #     steps=2000,
    #     hooks=[logging_hook]) # We pass our logging_hook to the hooks argument, so that it will be triggered during training.
    #
    # # Evaluate the model and print results
    # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": eval_data},
    #     y=eval_labels,
    #     num_epochs=10,
    #     shuffle=False)
    # eval_results = classifier.evaluate(input_fn=eval_input_fn)
    # print(eval_results)

if __name__ == "__main__":


  tf.app.run()
