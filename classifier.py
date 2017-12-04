from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

## Data sets
#DATA_TRAINING = "data_train.csv"
#DATA_TEST = "data_test.csv"

DATA_TRAINING = "train_2.csv"
DATA_TEST = "test_2.csv"


def main():
  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=DATA_TRAINING,
      target_dtype=np.float32,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=DATA_TEST,
      target_dtype=np.float32,
      features_dtype=np.float32)

  # Specify that all features have real-value data
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=10)]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[40,60,40],
                                              n_classes=2)
  # Define the training inputs
  def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)

    return x, y

  # Define the test inputs
  def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)

    return x, y

  for i in range(50):
      # Fit model.
      classifier.fit(input_fn=get_train_inputs, steps=500)

      # Evaluate accuracy.
      accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                           steps=1)["accuracy"]

      print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

if __name__ == "__main__":
    main()
