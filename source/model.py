import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from data import parser
# %matplotlib inline
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0" # for training on gpu

def make_one_hot(dataset):
    new_dataset = []
    for e in dataset:
        e = int(e)
        vec = np.zeros(2)
        vec[e] = 1;
        new_dataset.append(vec)
    return np.array(new_dataset)



# Create dictionary of target classes
label_dict = {
0: 'cage relevant',
1: 'cage irrelevant',
}

# plt.figure(figsize=[5,5])
#
# # Display the first image in training data
# plt.subplot(121)
# # curr_img = np.reshape(data.train.images[0], (28,28))
# curr_img = np.reshape(x_train[0], (28,28))
# # curr_lbl = np.argmax(data.train.labels[0,:])
# curr_lbl = y_train[0][0]
# print("curr_lbl:", curr_lbl)
# plt.imshow(curr_img, cmap="gray")
# print(label_dict[curr_lbl])
# plt.title("(Label:" + str(label_dict[curr_lbl]) + ")" )
#
# # Display the first image in testing data
# plt.subplot(122)
# # curr_img = np.reshape(data.test.images[0], (28,28))
# # curr_lbl = np.argmax(data.test.labels[0,:])
# curr_img = np.reshape(x_test[0], (28,28))
# curr_lbl = y_test[0][0]
# plt.imshow(curr_img, cmap="gray")
# plt.title("(Label:" + str(label_dict[curr_lbl]) + ")")

# plt.show()
#
#
# """ Data Preprocessing """
# print(data.train.images[0])
# print(np.max(data.train.images[1]))
# print(np.min(data.train.images[1]))
#
# # Reshpae training and testing image
# train_x = data.train.images.reshape(-1,28,28,1)
# test_x = data.test.images.reshape(-1,28,28,1)
# print(train_x.shape)
# print(test_x.shape)
# train_y = data.train.labels
# test_y = data.test.labels
# print(train_y.shape)
# print(test_y.shape)
#
# print("from fashion-mnist: ",x_train.shape)
# x_train = np.reshape(x_train,(-1,28,28,1))
# print(x_train.shape)

""" Hyperparameter setting"""
training_iters = 10
learning_rate = 0.001
batch_size = 128
# MNIST data input (img shape: 28*28)
n_input = 64 # in our case (img: 64*64)
n_classes = 2

# both placeholders are of type float
x = tf.placeholder("float", [None, 64,64,2])
y = tf.placeholder("float", [None, n_classes])

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x,b)

    return tf.nn.relu(x)

def maxpool2d(x, k=2):

    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

def dropout(x, keep_prob):

    return tf.nn.dropout(x, keep_prob)

weights = {
    'wc1': tf.get_variable('W0', shape=(7,7,2,64), initializer=tf.contrib.layers.xavier_initializer()), # original: shape=(3,3,1,32)
    'wc2': tf.get_variable('W1', shape=(5,5,64,64), initializer=tf.contrib.layers.xavier_initializer()),
    'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
    'wd1': tf.get_variable('W3', shape=(8*8*128,128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('W6', shape=(128, n_classes), initializer=tf.contrib.layers.xavier_initializer()),

}

biases = {
    'bc1': tf.get_variable('B0', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(2), initializer=tf.contrib.layers.xavier_initializer()),  # original: shape=(10)
}

keep_prob = tf.placeholder(tf.float32)

def conv_net(x, weights, biases):

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs as 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)
    conv2 = dropout(conv2, keep_prob)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv3 = maxpool2d(conv3, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Ouput, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# def input_fn(filename):
#     # Read in the training and test data in tfrecords format
#     filenames = tf.placeholder(tf.string, shape=[None])
#     dataset = tf.data.TFRecordDataset(filenames)
#
#     # Map the parser over dataset, and batch results by up to batch_size
#     dataset = dataset.map(parser,num_parallel_calls=None)
#     dataset = dataset.batch(batch_size)
#     dataset = dataset.repeat()
#     iterator1 = dataset.make_initializable_iterator()
#     iterator2 = dataset.make_initializable_iterator()
#     return dataset

global_step = tf.Variable(0, name='global_step', trainable=False)
pred = conv_net(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

# Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))

# calculate accuracy across all the given images and average them out
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()

training_dataset = "re_dataset_100_200.tfrecords"
test_dataset = "re_dataset_300_400.tfrecords"
# training_dataset = "small_dataset_100_200.tfrecords"
# test_dataset = "test_small_dataset_100_200.tfrecords"

with tf.Session() as sess:
    sess.run(init)
    # Restore variables from disk.
    # save_path = "/Users/chingandywu/GRASP/checkpoint_4/"
    save_path = "./checkpoint_4/"
    if os.path.exists(save_path):
        saver.restore(sess, tf.train.latest_checkpoint(save_path))
        print("Model restored.")

    step = global_step.eval(session=sess)
    print("#########################Global Step: ", step)

    # Read in the training and test data in tfrecords format
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames)

    # Map the parser over dataset, and batch results by up to batch_size
    # dataset = dataset.map(parser,num_parallel_calls=2) # depends on the number of cores of your cpu 
    # dataset = dataset.batch(batch_size)
    dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parser, batch_size=batch_size))
    dataset = dataset.prefetch(buffer_size=batch_size) # optimize the input pipeline
    dataset = dataset.repeat()
    iterator1 = dataset.make_initializable_iterator()
    iterator2 = dataset.make_initializable_iterator()


    # training_filenames = [os.path.join(DATASETNAME +'.tfrecords')]
    # training_dataset = "re_dataset_100_200.tfrecords"
    # training_dataset = "small_dataset_100_200.tfrecords"
    sess.run(iterator1.initializer, feed_dict={filenames:[training_dataset]})
    train_images, train_labels, filename,f1,f2 = iterator1.get_next()
    # test_dataset = "re_dataset_300_400.tfrecords"
    # test_dataset = "test_small_dataset_100_200.tfrecords"
    sess.run(iterator2.initializer, feed_dict={filenames:[test_dataset]})
    test_images, test_labels, filename,f1,f2 = iterator2.get_next()

    # """ check if training data is different from test data"""
    # if sess.run(tf.reduce_all(tf.equal(train_images, test_images))):
    #     print('train_images is equal test_images')
    # else:
    #     print('train_images is not equal test_images')


    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)


    for i in range(training_iters):

        # sess.run([train_op,increment_global_step],feed_dict=feed_dict)
        # if accuracy < 0.9:
            # batch_x = x_train[batch*batch_size:min((batch+1)*batch_size,len(x_train))]
            # batch_y = y_train[batch*batch_size:min((batch+1)*batch_size,len(y_train))]
            # # Run optimization op (backprop)
            # calculate batch loss and accuracy
        # traing data in a batch
        batch_x, batch_y = sess.run([train_images, train_labels])
        # data preprocessing
        batch_x = batch_x/255
        batch_y = make_one_hot(batch_y)

        opt = sess.run(optimizer, feed_dict={x:batch_x, y:batch_y, keep_prob: 0.8})
        loss, acc = sess.run([cost, accuracy], feed_dict={x:batch_x, y:batch_y, keep_prob:1.0})

        print("Iter "+ str(i + step + 1) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        print("Optimization Finished!")


        # test data in a batch
        x_test, y_test = sess.run([test_images, test_labels])
        # data preprocessing
        x_test = x_test/255
        y_test = make_one_hot(y_test)

        """ show training images and test images"""
        if i < 2:

            plt.subplot(221)
            img = (batch_x[0,:,:,0] + batch_x[0,:,:,1])/2.0
            plt.imshow(img)
            label = np.nonzero(batch_y[0,:])[0][0]
            plt.title("label: "+ label_dict[label])

            plt.subplot(222)
            img = (batch_x[1,:,:,0] + batch_x[1,:,:,1])/2.0
            plt.imshow(img)
            label = np.nonzero(batch_y[1,:])[0][0]
            plt.title("label: "+ label_dict[label])

            plt.subplot(223)
            img = (x_test[0,:,:,0] + x_test[0,:,:,1])/2.0
            plt.imshow(img)
            label = np.nonzero(y_test[0,:])[0][0]
            plt.title("label: "+ label_dict[label])

            plt.subplot(224)
            img = (x_test[1,:,:,0] + x_test[1,:,:,1])/2.0
            plt.imshow(img)
            label = np.nonzero(y_test[1,:])[0][0]
            plt.title("label: "+ label_dict[label])
            plt.tight_layout()
            plt.show()


        # print("batch_x: \n", type(batch_x), batch_x.shape)
        # print("batch_y: \n", type(batch_y), batch_y.shape)
        # print("x_test: \n", type(x_test), x_test.shape)
        # print("y_test: \n", type(y_test), y_test.shape)

        test_acc, valid_loss = sess.run([accuracy, cost], feed_dict={x:x_test, y: y_test, keep_prob: 1.0})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:", "{:.5f}".format(test_acc))



    summary_writer.close()
    save_path = saver.save(sess, "/Users/chingandywu/GRASP/checkpoint_4/", global_step=global_step)
    print("Model saved in path: %s" % save_path)

""" plot the training and test loss """
plt.plot(range(len(train_loss)), train_loss, 'b', label="Training loss")
plt.plot(range(len(train_loss)), test_loss, 'r', label="Test loss")
plt.title("Training and Test loss")
plt.xlabel("Epochs ", fontsize=16)
plt.ylabel("Loss ", fontsize=16)
plt.legend()
plt.show()

""" plot the trainng and test accuracy """
plt.plot(range(len(train_accuracy)), train_accuracy, 'b', label="Training accuracy")
plt.plot(range(len(test_accuracy)), test_accuracy, 'r', label="Test accuracy")
plt.title("Training and Test Accuracy")
plt.xlabel("Epochs ", fontsize=16)
plt.ylabel("Accuracy ", fontsize=16)
plt.legend()

plt.show()
