import sys
import os
import re
import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# dropped sample GSM1539667 from datasset because it was status "Other"

def get_csv_lines(data_file):
    infile = open(data_file)
    contents_raw = infile.read()
    infile.close()
    contents_split_nl = contents_raw.split("\n")
    return contents_split_nl

ROOT_PATH = "C:/Users/Nick/Documents/csc 450"

# training data lists
labels = []     # labels for data are sample status (0 = Control 1 = Alzheimer's)
values = []     # expression values (cells in the spreadsheet)
samples = []    # samples matched by ID to labels - each is list of expression values for that sample

# testing data lists
test_samples = []
test_labels = []
test_values = []


# Load gene expression training dataset and separate it into samples
expression_lines = get_csv_lines("expression_training.csv")

expression_lines_len = len(expression_lines)
line_len = len(expression_lines[0].split(","))

labels = expression_lines[1].split(",") # get labels from second row (first row is ID numbers that aren't used here)

for i in range(2, expression_lines_len - 1):    # split lines by commas
    comma_split_line = expression_lines[i].split(",")
    values.append(comma_split_line)

for i in range(0, line_len):                    # build samples
    tmp_list = []
    for j in range(0, expression_lines_len - 3): # -3 because first 2 rows were skipped
         tmp_list.append(values[j][i])
    samples.append(tmp_list)

unique_labels = set(labels) ### idk what this is for but it was in the example code?
print("LOAD/BUILD TRAINING SAMPLES OK")


# Load gene expression test dataset and separate it into samples
test_samples = []
test_labels = []
test_values = []


expression_lines = get_csv_lines("expression_testing.csv")

expression_lines_len = len(expression_lines)
line_len = len(expression_lines[0].split(","))

test_labels = expression_lines[1].split(",") # get labels from second row

for i in range(2, expression_lines_len - 1):    # split lines by commas
    comma_split_line = expression_lines[i].split(",")
    test_values.append(comma_split_line)

for i in range(0, line_len):                    # build samples
    tmp_list = []
    for j in range(0, expression_lines_len - 3): # -3 because first 2 rows were skipped
         tmp_list.append(test_values[j][i])
    test_samples.append(tmp_list)

unique_test_labels = set(test_labels)



print("LOAD/BUILD TEST SAMPLES OK")

x = tf.placeholder(dtype = tf.float32, shape = [None, 1616])    # 1616 = number of values per sample
y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten the input data (this still seems to be necessary to do the fully_connected call below?
samples_flat = tf.contrib.layers.flatten(x)

# Fully connected layer
logits = tf.contrib.layers.fully_connected(samples_flat, 2, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                    logits = logits))
# Define an optimizer 
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print("TF SETUP OK")

print("samples_flat: ", samples_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)

tf.set_random_seed(1234)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(201):
        print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: samples, y: labels})
        if i % 10 == 0:
            print("Loss: ", loss)
        print('DONE WITH EPOCH')

sample_indexes = random.sample(range(len(samples)), 10)
sample_samples = [samples[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "predicted_labels" op.
predicted = sess.run([correct_pred], feed_dict={x: sample_samples})[0]
                        
# Print the real and predicted labels
print(sample_labels)
print(predicted)

predicted = sess.run([correct_pred], feed_dict={x: test_samples})[0]

# Calculate correct matches 
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

# Calculate the accuracy
accuracy = match_count / len(test_labels)

# Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))

print("END")




sys.exit()


############ PROGRAM ENDS HERE, CODE BELOW IS JUST HERE FOR REFERENCE




### WORKING EXAMPLE TAKEN FROM https://www.datacamp.com/community/tutorials/tensorflow-tutorial
### Above code is based on this

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "C:/Users/Nick/Documents/csc 450"
train_data_directory = os.path.join(ROOT_PATH, "Training")
test_data_directory = os.path.join(ROOT_PATH, "Testing")

images, labels = load_data(train_data_directory)
print("Load OK")

unique_labels = set(labels)

# Rescale the images in the `images` array
arr_images = np.array(images)
images28 = [transform.resize(image, (28, 28)) for image in arr_images]
images28 = np.array(images28)
images28 = rgb2gray(images28)

print ("IMAGES OK")

# Initialize placeholders 
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer 
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                    logits = logits))
# Define an optimizer 
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print("TF SETUP OK")

print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)

tf.set_random_seed(1234)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(201):
        print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
        if i % 10 == 0:
            print("Loss: ", loss)
        print('DONE WITH EPOCH')

sample_indexes = random.sample(range(len(images28)), 10)
sample_images = [images28[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "predicted_labels" op.
predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
                        
# Print the real and predicted labels
print(sample_labels)
print(predicted)

print("SAMPLE PREDICT OK")

test_images, test_labels = load_data(test_data_directory)

# Transform the images to 28 by 28 pixels
test_images28 = [transform.resize(image, (28, 28)) for image in test_images]

# Convert to grayscale
test_images28 = rgb2gray(np.array(test_images28))

# Run predictions against the full test set.
predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]

# Calculate correct matches 
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

# Calculate the accuracy
accuracy = match_count / len(test_labels)

# Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))

print("END")


#### END EXAMPLE


#### DATA PREP STUFF
def get_id_status_dict(data_file):
    infile = open(data_file)
    contents_raw = infile.read()
    infile.close()

    contents_split_nl = contents_raw.split("\n")

    d = {}

    for the_line in contents_split_nl:
        line_split_comma = the_line.split(",")
        d[line_split_comma[0]] = line_split_comma[10]  

    return d

def get_status_list(data_file):
    infile = open(data_file)
    contents_raw = infile.read()
    infile.close()

    contents_split_nl = contents_raw.split("\n")

    the_list = list()

    print(contents_split_nl[0])
    print(contents_split_nl[1])

    i = 0

    for the_line in contents_split_nl:
        line_split_comma = the_line.split(",")
        if (len(line_split_comma) > 10):
            the_list.append(line_split_comma[10])
            print(str(i) + " : " + line_split_comma[0] + " : " + line_split_comma[10])
            i += 1
    return the_list


status_list = get_status_list("Nick_sample_info_mod.csv")
print(status_list)

expression_lines = get_csv_lines("Nick_expression_mod.csv")

outfile = open("full_expression.csv","w")

status_line = ""

for i in range(0, len(status_list)) :
    status_line += status_list[i] + ","

status_line = status_line[0:-1:]

outfile.write(expression_lines[0] + "\n")
outfile.write(status_line + "\n")

for i in range(1, len(expression_lines)) :
    outfile.write(expression_lines[i] + "\n")



outfile.close()
