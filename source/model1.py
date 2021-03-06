import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from data import parser
# %matplotlib inline
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0" # for training on gpu


TRAIN =  True # if you want to train the model, set this parameter equals to True.
""" DATA SET SETTING """
training_dataset_1 = "/Users/chingandywu/GRASP/tf-dataset/re_dataset_0_100_train.tfrecords"
training_dataset_2 = "/Users/chingandywu/GRASP/tf-dataset/re_dataset_100_200_train.tfrecords"
training_dataset_3 = "/Users/chingandywu/GRASP/tf-dataset/re_dataset_300_400_train.tfrecords"
training_dataset_4 = "/Users/chingandywu/GRASP/tf-dataset/re_dataset_500_600_train.tfrecords"
training_dataset_5 = "/Users/chingandywu/GRASP/tf-dataset/re_train_dataset_600_700.tfrecords"
training_dataset_6 = "/Users/chingandywu/GRASP/tf-dataset/re_train_dataset_700_800.tfrecords"

test_dataset_1 = "/Users/chingandywu/GRASP/tf-dataset/re_dataset_0_100_test.tfrecords"
test_dataset_2 = "/Users/chingandywu/GRASP/tf-dataset/re_dataset_100_200_test.tfrecords"
test_dataset_3 = "/Users/chingandywu/GRASP/tf-dataset/re_dataset_300_400_test.tfrecords"
test_dataset_4 = "/Users/chingandywu/GRASP/tf-dataset/re_test_dataset_600_700.tfrecords"

evaluate_dataset = "/Users/chingandywu/GRASP/tf-dataset/re_dataset_100_200_test.tfrecords"
# evaluate_dataset = "/Users/chingandywu/GRASP/tf-dataset/re_Dataset_test.tfrecords"
# evaluate_dataset = "/Users/chingandywu/GRASP/tf-dataset/re_dataset_400_500.tfrecords"
restore_path = "./folder/"
# restore_path = "./checkpoint_test/"
save_path = restore_path
fig_folder = 'folder'
# fig_folder = 'fig_test'

""" Hyperparameter setting"""
epochs = 3
learning_rate = 0.001
# batch_size = 128
batch_size = 128



n_input = 64 # in our case (img: 64*64)
n_classes = 2
def make_one_hot(dataset):
    new_dataset = []
    for e in dataset:
        e = int(e)
        vec = np.zeros(2)
        vec[e] = 1;
        new_dataset.append(vec)
    return np.array(new_dataset)

def save_images(train_loss, test_loss, train_accuracy, test_accuracy, global_step):

    """ plot the training and test loss """
    plt.plot(train_loss, 'b', label="Training loss")
    plt.plot(test_loss, 'r', label="Test loss")
    # plt.plot(range(len(train_loss)), train_loss, 'b', label="Training loss")
    # plt.plot(range(len(train_loss)), test_loss, 'r', label="Test loss")
    plt.title("Training and Test loss")
    plt.xlabel("Steps ", fontsize=16)
    plt.ylabel("Loss ", fontsize=16)
    plt.legend()
    plt.savefig('../'+ fig_folder +'/loss-' + str(global_step) + '.png')
    plt.clf()


    """ plot the trainng and test accuracy """
    plt.plot(train_accuracy, 'b', label="Training accuracy")
    plt.plot(test_accuracy, 'r', label="Test accuracy")
    # plt.plot(range(len(train_accuracy)), train_accuracy, 'b', label="Training accuracy")
    # plt.plot(range(len(test_accuracy)), test_accuracy, 'r', label="Test accuracy")
    plt.title("Training and Test Accuracy")
    plt.xlabel("Steps ", fontsize=16)
    plt.ylabel("Accuracy ", fontsize=16)
    plt.legend()
    plt.savefig('../'+ fig_folder +'/accuracy-' + str(global_step) + '.png')
    plt.clf()



# Create dictionary of target classes
label_dict = {
0: 'cage relevant',
1: 'cage irrelevant',
}


# both placeholders are of type float
x = tf.placeholder("float", [None, 64,64,2])
y = tf.placeholder("float", [None, n_classes])

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x,b)

    return tf.nn.relu(x)

def maxpool2d(x, k=2):

    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='VALID')

def dropout(x, keep_prob):

    return tf.nn.dropout(x, keep_prob)

def lrn(x):
    return tf.nn.local_response_normalization(x, bias=2, alpha=1e-4, beta=0.75)

def relu(x):
    return tf.nn.relu(x)

weights = {
    'wc1': tf.get_variable('W0', shape=(5,5,2,32), initializer=tf.contrib.layers.xavier_initializer()), # original: shape=(3,3,1,32)
    'wc2': tf.get_variable('W1', shape=(5,5,32,64), initializer=tf.contrib.layers.xavier_initializer()),
    # 'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
    'wd1': tf.get_variable('W3', shape=(16*16*64,1024), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('W6', shape=(1024, n_classes), initializer=tf.contrib.layers.xavier_initializer()),

}

biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    # 'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(1024), initializer=tf.contrib.layers.xavier_initializer()),
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
    # conv2 = dropout(conv2, keep_prob)

    # conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    # conv3 = maxpool2d(conv3, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = dropout(fc1, keep_prob)

    # Ouput, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

global_step = tf.Variable(0, name='global_step', trainable=False)
pred = conv_net(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

# Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector
# y_pred = tf.argmax(pred,1)
# y_true = tf.argmax(y,1)
# correct_prediction = tf.equal(y_pred, y_true)
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))

# calculate accuracy across all the given images and average them out
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()

# Read in the training and test data in tfrecords format
filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.shuffle(buffer_size = batch_size)
# Map the parser over dataset, and batch results by up to batch_size
# dataset = dataset.map(parser,num_parallel_calls=2) # depends on the number of cores of your cpu
# dataset = dataset.batch(batch_size)
dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parser, batch_size=batch_size))
dataset = dataset.prefetch(buffer_size=batch_size) # optimize the input pipeline
# dataset = dataset.repeat()

iterator = dataset.make_initializable_iterator()
images, labels, filename,f1,f2 = iterator.get_next()

#
# train_images, train_labels, filename,f1,f2 = iterator.get_next()
#
# iterator_2= dataset_2.make_initializable_iterator()
# test_images, test_labels, filename,f1,f2 = iterator_2.get_next()
def main():



    with tf.Session() as sess:
        sess.run(init)
        # Restore variables from disk.
        # save_path = "/Users/chingandywu/GRASP/checkpoint_4/"
        # save_path = "./checkpoint_0/"
        if os.path.exists(save_path):
            saver.restore(sess, tf.train.latest_checkpoint(save_path))
            print("Model restored.")

        step = global_step.eval(session=sess)
        print("############### Global Step: ", step, "###################")


        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []
        summary_writer = tf.summary.FileWriter('./output', sess.graph)

        # pre_batch = 0
        for i in range(epochs):
            print("Epoch: ", i)
            count = 0


            """ Training Phase """
            # initialze training iterator
            sess.run(iterator.initializer, feed_dict={filenames:[training_dataset_1,training_dataset_2,training_dataset_3,training_dataset_4, training_dataset_5, training_dataset_6]})
            while True:
            # for iter in range(20):
                try:
                    # traing data in a batch
                    batch_x, batch_y, file_train= sess.run([images, labels, filename])

                    # data preprocessing
                    batch_x = batch_x/255
                    batch_y = make_one_hot(batch_y)

                    opt = sess.run(optimizer, feed_dict={x:batch_x, y:batch_y, keep_prob: 0.6}) # chinge the dropout rate here
                    loss, acc = sess.run([cost, accuracy], feed_dict={x:batch_x, y:batch_y, keep_prob:1.0})
                    train_accuracy.append(acc)
                    train_loss.append(loss)
                    # if np.all(pre_batch == batch_x):
                    #     print("#"*50)
                    #     print("The same")
                    # else:
                    #     print("#"*50)
                    #     print("Different!!!!")
                    # pre_batch = batch_x
                    print("  batch: ", count)
                    if count % 50 == 0:
                        path = saver.save(sess, save_path, global_step=global_step)
                        print("Model saved in path: %s" % path)
                    count += 1
                except tf.errors.OutOfRangeError:
                    break

            """ Validation Phase"""
            # initialize test iterator
            sess.run(iterator.initializer, feed_dict={filenames:[test_dataset_1, test_dataset_2, test_dataset_3, test_dataset_4]})
            while True:
            # for iter in range(20):
                try:
                    # test data in a batch
                    x_test, y_test, file_test = sess.run([images, labels, filename])
                    # data preprocessing
                    x_test = x_test/255
                    y_test = make_one_hot(y_test)
                    valid_acc, valid_loss = sess.run([accuracy, cost], feed_dict={x:x_test, y: y_test, keep_prob: 1.0})
                    test_loss.append(valid_loss)
                    test_accuracy.append(valid_acc)
                    if valid_acc < np.min(test_accuracy[-10:]):
                        print("Min validation accracy: ", np.min(test_accuracy[-10:]))
                        print("Current valid accuracy: ", valid_acc)
                        s = sess.run(global_step)
                        save_images(train_loss, test_loss, train_accuracy, test_accuracy, s)
                        quit()
                except tf.errors.OutOfRangeError:
                    break

            # train_loss.append(loss)
            # test_loss.append(valid_loss)
            # train_accuracy.append(acc)
            # test_accuracy.append(test_acc)
            print("-"*50)
            print("filename train: \n", file_train)
            print("-"*50)
            print("filename test: ", file_test)

            # print("Iter "+ str(i + step + 1) + ", Loss= " + "{:.6f}".format(loss))
            # print("Training Accuracy= " + "{:.5f}".format(acc) + "Testing Accuracy:", "{:.5f}".format(test_acc))
            # print("Iter "+ str(i + step + 1) + ", Loss= " + "{:.6f}".format(loss))
            max_train_acc = np.max(train_accuracy)
            max_valid_acc = np.max(test_accuracy)
            print("Training Accuracy= " + "{:.5f}".format(max_train_acc) + "Testing Accuracy:", "{:.5f}".format(max_valid_acc))

            s = sess.run(global_step)
            save_images(train_loss, test_loss, train_accuracy, test_accuracy, s)


        summary_writer.close()
        path = saver.save(sess, save_path, global_step=global_step)
        print("Model saved in path: %s" % path)

    """ plot the training and test loss """
    plt.figure()
    plt.plot(train_loss, 'b', label="Training loss")
    plt.plot(test_loss, 'r', label="Test loss")
    plt.title("Training and Test loss")
    plt.xlabel("Epochs ", fontsize=16)
    plt.ylabel("Loss ", fontsize=16)
    plt.legend()
    plt.savefig('../' + fig_folder + '/loss_last.png')
    plt.show()

    """ plot the training and test accuracy"""
    plt.figure()
    plt.plot(train_accuracy, 'b', label="Training accuracy")
    plt.plot(test_accuracy, 'r', label="Test accuracy")
    plt.title("Training and Test Accuracy")
    plt.xlabel("Epochs ", fontsize=16)
    plt.ylabel("Accuracy ", fontsize=16)
    plt.legend()
    plt.savefig('../' + fig_folder+ '/accuracy_last.png')
    plt.show()

def evaluate():

    with tf.Session() as sess:
        sess.run(init)
        # Restore variables from disk.
        # save_path = "/Users/chingandywu/GRASP/checkpoint_4/"
        # save_path = "./checkpoint_0/"
        if os.path.exists(save_path):
            saver.restore(sess, tf.train.latest_checkpoint(save_path))
            print("#"*80)
            print("Model restored.")

        step = global_step.eval(session=sess)
        print("############### Global Step: ", step, "###################")

        # Read in the training and test data in tfrecords format
        filenames = tf.placeholder(tf.string, shape=[None])
        dataset = tf.data.TFRecordDataset(filenames)
        # Map the parser over dataset, and batch results by up to batch_size
        # dataset = dataset.map(parser,num_parallel_calls=2) # depends on the number of cores of your cpu
        # dataset = dataset.batch(batch_size)
        dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parser, batch_size=batch_size))
        dataset = dataset.prefetch(buffer_size=batch_size) # optimize the input pipeline
        dataset = dataset.repeat()
        # iterator1 = dataset.make_initializable_iterator()
        iterator = dataset.make_initializable_iterator()


        # sess.run(iterator.initializer, feed_dict={filenames:[training_dataset]})
        # train_images, train_labels, filename,f1,f2 = iterator.get_next()

        sess.run(iterator.initializer, feed_dict={filenames:[evaluate_dataset]})
        test_images, test_labels, filename,f1,f2 = iterator.get_next()

        # train_loss = []
        test_loss = []
        # train_accuracy = []
        test_accuracy = []
        summary_writer = tf.summary.FileWriter('./output', sess.graph)
        f1_list = []
        precision_list = []

        # for i in range(steps):
        while True:
            try:
                # test data in a batch
                x_test, y_test = sess.run([test_images, test_labels])
                # data preprocessing
                x_test = x_test/255
                y_test = make_one_hot(y_test)
                # valid_acc, valid_loss = sess.run([accuracy, cost], feed_dict={x:x_test, y: y_test, keep_prob: 1.0})
                y_p = tf.argmax(pred,1)
                acc, loss, y_pred = sess.run([accuracy, cost, y_p], feed_dict={x:x_test, y: y_test, keep_prob: 1.0})
                test_loss.append(loss)
                test_accuracy.append(acc)
                # test_loss.append(valid_loss)
                # test_accuracy.append(test_acc)

                y_true = np.argmax(y_test,1)
                TP = tf.count_nonzero(y_pred * y_true)
                FP = tf.count_nonzero(y_pred * (y_true - 1))
                FN = tf.count_nonzero((y_pred - 1) * y_true)

                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = 2 * precision * recall / (precision + recall)

                precision = sess.run(precision)
                f1 = sess.run(f1)
                precision_list.append(precision)
                f1_list.append(f1)

                print("TP: ", sess.run(TP))
                print("FP: ", sess.run(FP))
                print("FN: ", sess.run(FN))
                print("precision: ", precision)
                print("F1 score: ", f1)
                print("Testing Accuracy:", "{:.5f}".format(acc))
                # print("y_true: \n", y_true)
                # print("y_pred: \n", y_pred)
            except tf.errors.OutOfRangeError:
                break

        print("Max Test Accuracy: ", np.max(test_accuracy))
        print("Avg Test Accuracy: ", np.average(test_accuracy))

        print("Max Precision: ", np.max(precision_list))
        print("Avg Precision: ", np.average(precision_list))

        print("Max f1 score: ", np.max(f1_list))
        print("Avg f1 score: ", np.average(f1_list))
        # summary_writer.close()
        # path = saver.save(sess, save_path, global_step=global_step)
        # print("Model saved in path: %s" % path)

if __name__ == '__main__':
    if TRAIN == True:
        main()
    else:
        steps = 10
        evaluate()
