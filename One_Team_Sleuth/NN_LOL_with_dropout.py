# Kenneth Hall
# Neural Network win classifier using past game data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os


curr_dir = os.getcwd()
data_dir = os.path.join(curr_dir, 'data')

input_train_path = os.path.join(data_dir, 'train.csv')

input_test_path = os.path.join(data_dir, 'test.csv')


FLAGS = None
sample_length = 18

disp_analysis = True

# Parameters
learning_rate = .0001
train_steps = 10000
dropout_ratio = 0.5
batch_size = 50

def lolnn(x):
    """lolnn builds the graph to perform match prediction with a neural network

    Args:
        x: input tensor with dimensions (num_samples, sample_length).

    Returns:
        A tensor of length two with decimal predictions for win vs. loss
    """

    # First fully connected layer
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([18, 50])
        b_fc1 = bias_variable([50])

        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    # Second fully connected layer
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([50,100])
        b_fc2 = bias_variable([100])

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # Third fully connected layer
    with tf.name_scope('fc3'):
        W_fc3 = weight_variable([100,20])
        b_fc3 = bias_variable([20])

        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

    # Dropout - minimizes overfitting
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

    # Third fully connected layer
    with tf.name_scope('fc4'):
        W_fc4 = weight_variable([20,2])
        b_fc4 = bias_variable([2])

        h_fc4 = tf.matmul(h_fc3_drop, W_fc4) + b_fc4

    return h_fc4, keep_prob


# Define how we create weight variables and bias variables
def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def build_basic_graph():
    # Input tensor
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 18])

    # Build the FFN
    y_pred, keep_prob = lolnn(x)

    return x, y_pred, keep_prob


def build_training_graph(learn, pred):
    # True y tensor (targets, or labels)
    with tf.name_scope('correct'):
        y_true = tf.placeholder(tf.float32, [None,2])

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=pred)
        loss = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(learn).minimize(loss)
    return y_true, loss, train_step


# generators will loop forever if batch_size > samples, also it has the chance to miss a
# few samples each iteration, though they all have equal probability, so it shouldnt matter
def data_generator(data, size):
    np.random.shuffle(data)
    sample_length = len(data)
    curr = sample_length
    loop = 0
    while True:
        if curr+size > sample_length:
            curr = 0
            np.random.shuffle(data)
            loop += 1
            print('looping training data for the {0} time'.format(loop))
            continue
        x = data[curr:curr+size, :18]
        y_labels = data[curr:curr+size, 18]
        y_labels_onehot = []
        for label in y_labels:
            if int(label) == 1:
                y_labels_onehot.append([1,0])
            else:
                y_labels_onehot.append([0,1])
        curr += size
        yield x, y_labels_onehot


def get_data(size):
    # Import data and goal output data
    ## Training
    signal_data = np.loadtxt(open(input_train_path, 'r'), delimiter=',')
    all_train = signal_data
    #all_train = signal_data.swapaxes(0, 1)
    print('Number of training examples', len(all_train))

    ## Testing
    signal_data_test = np.loadtxt(open(input_test_path, 'r'), delimiter=',')
    all_test = signal_data_test
    #all_test = signal_data_test.swapaxes(0, 1)
    print('Number of test examples', len(all_test))

    #assert(all_train.shape[0] >= size)
    #assert(all_test.shape[0] >= size)
    return data_generator(all_train, size), data_generator(all_test, size)

def check_accuracy(y_pred, y_labels):
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_labels, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    return accuracy

def main():
    # data_imports
    train_gen, test_gen = get_data(batch_size)

    # build graph
    x, y_pred, keep_prob = build_basic_graph()
    y_true, loss, train_step = build_training_graph(learning_rate, y_pred)

    accuracy = check_accuracy(y_pred, y_true)

    # training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(train_steps):
            signal_data, y_labels = next(train_gen)
            train_step.run(feed_dict = {x: signal_data, y_true: y_labels,
                keep_prob: dropout_ratio})

            loss_step = sess.run(loss, feed_dict={x: signal_data, y_true: y_labels,
                keep_prob: 1.0})
            #y_pred = sess.run(fc4, feed_dict={x: data, y_true: y_labels, keep_prob: 1.0})
            accuracy_report = accuracy.eval(feed_dict={x: signal_data, y_true: y_labels,
                keep_prob: 1.0})
            print('Step %i: Loss: %f Accuracy: %f' % (i, loss_step, accuracy_report))
            #print('Step %i: Loss: %f' % (i, loss_step))

            # disp_analysis = False
            if disp_analysis and i % 2000 == 0:
                # Test step
                signal_data_test, y_labels = next(test_gen)
                l = sess.run(loss, feed_dict={x: signal_data_test,
                    y_true: y_labels, keep_prob: 1.0})
                #y_pred = sess.run(fc4,feed_dict={x: data, y_true: y_labels, keep_prob: 1.0})
                accuracy_report = accuracy.eval(feed_dict={x: signal_data, y_true: y_labels,
                    keep_prob: 1.0})
                print('Testing step %i: Loss: %f Accuracy: %f' % (i, l, accuracy_report))
                #print('Step %i: Loss: %f' % (i, l))
                input("Press Enter to continue...")
if __name__ == "__main__":
    main()
