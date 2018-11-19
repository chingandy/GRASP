from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    
    print("This is labels shape", labels.get_shape())
    print("This is features shape ", features)
    #Input layer
    input_layer = tf.reshape(features["image"], [-1,64,64,2])
    print("This is the shape pf input ", input_layer.get_shape)

    #First convolutional layer
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 32,
        kernel_size = [5,5],
        padding = "same",
        activation = tf.nn.relu
    )

    #Max pooling layer 1
    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2,2], strides = 2)

    #Second convolutional layer
    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 64,
        kernel_size = [5,5],
        padding = "same",
        activation = tf.nn.relu
    )

    #Max pooling layer 2
    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2,2], strides = 2)


    pool2_flat = tf.reshape(pool2, [-1, 16 * 16 * 64])
    dense = tf.layers.dense(inputs = pool2_flat, units = 1024, activation = tf.nn.relu)
    dropout = tf.layers.dropout(inputs = dense, rate = 0.4, training=mode == tf.estimator.ModeKeys.TRAIN) 

    logits = tf.layers.dense(inputs = dropout, units = 2)
    print("The shape of logits ", logits.get_shape)    
    predictions = {
        "classes": tf.argmax(input = logits, axis = 1),
        "probabilities": tf.nn.softmax(logits, name = "softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(
            loss = loss,
            global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)
    
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels = labels, predictions = predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)

def parser(record):
    keys_to_feature = {
        "image/encoded64": tf.FixedLenFeature([], tf.string),
        "image/channels": tf.FixedLenFeature([], tf.int64),
        "image/trainlable": tf.FixedLenFeature([], tf.float32),
        "image/name": tf.FixedLenFeature([], tf.string),
        "image/f1": tf.FixedLenFeature([], tf.string),
        "image/f2": tf.FixedLenFeature([], tf.string)
    }
    parsed = tf.parse_single_example(record, keys_to_feature)
    
    label = tf.cast(tf.reshape(parsed["image/trainlable"], shape = []), tf.int32)

    image_encoded64 = parsed["image/encoded64"]
    image_raw64 = tf.decode_raw(image_encoded64, tf.uint8)
    image64 = tf.reshape(image_raw64, [64,64,2])
    image64 = tf.cast(image64, tf.float32)
    print(label.get_shape())
    return {"image": image64}, label

def dataset_input_fn(batch_size, num_of_epochs):
    datasetname = ""
    #filenames = [os.path.join(dataasetname + 'tfrecords')]
    dataset = tf.data.TFRecordDataset("../../Data_science_dataset/Dataset_all_test.tfrecords")
    dataset = dataset.map(parser)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_of_epochs)
    iterator = dataset.make_one_shot_iterator() 
    features, labels = iterator.get_next()
    return features, labels

def main(unused_argv):
    
    #Parameters
    batch_size = 10 
    num_of_epochs = 1

    #Create estimator
    estimator = tf.estimator.Estimator(model_fn = cnn_model_fn, model_dir = "model_checkpoint")
    #print(estimator)

    #Logging hook
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    #Training specification
    train_spec = tf.estimator.TrainSpec(
        input_fn = lambda: dataset_input_fn(batch_size, num_of_epochs),
        max_steps = 500,
        hooks = [logging_hook]
    )
    
    #Evaluation specification
    eval_spec = tf.estimator.EvalSpec(
        input_fn = lambda: dataset_input_fn(batch_size, num_of_epochs),
        steps = None,
        name = None
    )

    #Train and evaluate
    result = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    print(result)

if __name__ == '__main__':
    tf.app.run()


