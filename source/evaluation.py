import tensorflow as tf
from model2 import *

iters = 2
test_dataset = "/Users/chingandywu/GRASP/tf-dataset/re_dataset_100_200.tfrecords"
# restore_path = "./ckeckpoint_unseen/"
restore_path = "/Users/chingandywu/GRASP/scr/checkpoint_unseen"
# restore_path = "/Users/chingandywu/GRASP/report/model2_unseen"


# global_step = tf.Variable(0, name='global_step', trainable=False)
# pred = conv_net(_, weights, biases)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

# Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector

# correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
#
# # calculate accuracy across all the given images and average them out
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# # Initializing the variables
# init = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
# y_pred = tf.argmax(pred,1)
# y_true = tf.argmax(y,1)
# f1_score = tf.contrib.metrics.f1_score(y_true, y_pred) # calculate f1 score
batch_size = 512

with tf.Session() as sess:
    sess.run(init)
    # Restore variables from disk.
    # save_path = "/Users/chingandywu/GRASP/checkpoint_4/"
    # save_path = "./checkpoint_0/"
    if os.path.exists(restore_path):
        saver.restore(sess, tf.train.latest_checkpoint(restore_path))
        print("Model restored.")
    else:
        print("Unable to restore the model!!!!")

    step = global_step.eval(session=sess)
    print("############Global Step: ", step, "############")

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
    #
    # sess.run(iterator.initializer, feed_dict={filenames:[test_dataset]})
    # test_images, test_labels, filename,f1,f2 = iterator.get_next()

    sess.run(iterator.initializer, feed_dict={filenames:[test_dataset]})
    test_images, test_labels, filename,f1,f2 = iterator.get_next()


    # train_loss = []
    test_loss = []
    # train_accuracy = []
    test_accuracy = []



    for i in range(iters):

        if i % 10 == 0:
            print("Evaluate Percentage: ", i/iters * 100," %")

        # test data in a batch
        x_test, y_test = sess.run([test_images, test_labels])
        # data preprocessing
        x_test = x_test/255
        y_test = make_one_hot(y_test)

        y_p = tf.argmax(pred,1)
        test_acc, valid_loss, y_pred = sess.run([accuracy, cost, y_p], feed_dict={x:x_test, y: y_test, keep_prob: 1.0})

        test_loss.append(valid_loss)
        test_accuracy.append(test_acc)

        y_true = np.argmax(y_test,1)
        TP = tf.count_nonzero(y_pred * y_true)
        FP = tf.count_nonzero(y_pred * (y_true - 1))
        FN = tf.count_nonzero((y_pred - 1) * y_true)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        # f1_score, best_op = tf.contrib.metrics.f1_score(y_true, y_pred)
        # f1_score= sess.run(f1_score)
        print("TP: ", sess.run(TP))
        print("FP: ", sess.run(FP))
        print("FN: ", sess.run(FN))
        print("precision: ", sess.run(precision))
        print("F1 score: ", sess.run(f1))
        print("Testing Accuracy:", "{:.5f}".format(test_acc))
        print("y_true: \n", y_true)
        print("y_pred: \n", y_pred)
        # if i % 10 == 0:
        #     save_images(_, test_loss, _, test_accuracy, i)


    # path = saver.save(sess, save_path, global_step=global_step)
    # print("Model saved in path: %s" % path)
""" plot the test loss and accuracy """
# plt.plot(range(len(test_acc)), test_acc, 'b', label="Test accuracy")
# plt.plot(range(len(test_loss)), test_loss, 'r', label="Test loss")
plt.figure()
# plt.plot(test_acc, 'b', label="Test accuracy")
plt.plot(test_loss, 'r', label="Test loss")
plt.plot(test_accuracy, 'b', label="Test accuracy")
plt.title("Test accuracy and loss")
plt.xlabel("Epochs ", fontsize=16)
plt.ylabel("Scale ", fontsize=16)
plt.legend()
plt.savefig('./test.png')
plt.show()

# """ plot the training and test loss """
# plt.plot(range(len(train_loss)), train_loss, 'b', label="Training loss")
# plt.plot(range(len(train_loss)), test_loss, 'r', label="Test loss")
# plt.title("Training and Test loss")
# plt.xlabel("Epochs ", fontsize=16)
# plt.ylabel("Loss ", fontsize=16)
# plt.legend()
# plt.savefig('fig2/loss.png')
# plt.show()
#
#
# """ plot the trainng and test accuracy """
# plt.plot(range(len(train_accuracy)), train_accuracy, 'b', label="Training accuracy")
# plt.plot(range(len(test_accuracy)), test_accuracy, 'r', label="Test accuracy")
# plt.title("Training and Test Accuracy")
# plt.xlabel("Epochs ", fontsize=16)
# plt.ylabel("Accuracy ", fontsize=16)
# plt.legend()
# plt.savefig('fig2/accuracy.png')
# plt.show()
