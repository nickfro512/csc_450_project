# Implements neural network which is trained on gene expression data to
# try and classify samples as Alzheimer's or Control. Part of CSC 450
# research project.
#
# Authors: Nick Frogley and Sohrob Kazzerounian

import sys
import os
import re
import random

import tensorflow as tf
import numpy as np

# dropped sample GSM1539667 from datasset because it was status "Other"

# read gene expression samples from CSV files into arrays 
def get_data(fname):

    with open(fname, 'r') as f:
        file_data = f.readlines()

    feature_names = file_data[0].strip().split(',')
    labels = [int(x) for x in file_data[1].strip().split(',')]
    inputs = []

    for samp in file_data[2:-1]:
        inputs.append([float(x) for x in samp.strip().split(',')])

    inputs = np.asarray(inputs).T

    return inputs, labels, feature_names

test_x, test_y, _ = get_data("expression_testing.csv")
train_x, train_y, _ = get_data("expression_training.csv")

print("LOAD/BUILD TRAINING SAMPLES OK")

print("LOAD/BUILD TEST SAMPLES OK")
print("Building Network...")

n_inputs = train_x.shape[0]
n_features = train_x.shape[1]
n_classes = len(set(train_y))

n_inputs_test = test_x.shape[0]

n_hidden_1 = 512
n_hidden_2 = 512

# tf Graph input
X = tf.placeholder("float", [None, n_features])
Y = tf.placeholder("int64", [None])

# add first hidden layer with dropout
h1 = tf.contrib.layers.fully_connected(X, n_hidden_1, tf.nn.relu)
h1_drop = tf.nn.dropout(h1, keep_prob=1)

# add second hidden layer with dropout
h2 = tf.contrib.layers.fully_connected(h1_drop, n_hidden_2, tf.nn.relu)
h2_drop = tf.nn.dropout(h2, keep_prob=1)

# calculate output layer logits
logits = tf.contrib.layers.fully_connected(h2_drop, n_classes, activation_fn=None)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,
                                                                     logits=logits))
# Define an optimizer 
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
#train_op = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)


# Convert logits to label indexes
predicted_classes = tf.argmax(logits, 1)
predicted_correct = tf.equal(predicted_classes, Y)

# set up true/false positive/negative values
fp = tf.metrics.false_positives(Y, predicted_classes)
fn = tf.metrics.false_negatives(Y, predicted_classes)
tp = tf.metrics.true_positives(Y, predicted_classes)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(predicted_correct, tf.float32))

print("TF SETUP OK")

# train the network
with tf.Session() as sess:
    
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # run 2000 epochs
    for i in range(2000):

        _, acc_val, loss_val = sess.run([train_op, accuracy, loss], feed_dict={X: train_x, Y: train_y})

        if i % 20 == 0:
           
            # Run the "predicted_labels" op.
            acc_val, logit_val = sess.run([accuracy, logits], feed_dict={X: test_x, Y: test_y})
            print("{:.3f}".format(acc_val))

    # get true/false positive/negative info and print it
    false_pos, false_neg, true_pos = sess.run([fp, fn, tp], feed_dict={X: test_x, Y: test_y})
    true_neg = n_inputs_test - (false_pos[0] + false_neg[0] + true_pos[0])
    print("FP: {}   FN: {}   TP: {}   TN: {}".format(false_pos[0], false_neg[0], true_pos[0], true_neg))
    

