import tensorflow as tf
from tensorflow.contrib import learn
import sys
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = '/tmp/data' if not 'win32' in sys.platform else "c:\\tmp\\data"
data = input_data.read_data_sets(DATA_DIR, one_hot=False)
x_data, y_data = data.train.images, data.train.labels.astype(np.int32)
x_test, y_test = data.test.images, data.test.labels.astype(np.int32)

NUM_STEPS = 2000
MINIBATCH_SIZE = 128

feature_columns = learn.infer_real_valued_columns_from_input(x_data)

dnn = learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[200], n_classes=10,
                          optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.2))

dnn.fit(x=x_data, y=y_data, steps=NUM_STEPS, batch_size=MINIBATCH_SIZE)

test_acc = dnn.evaluate(x=x_test, y=y_test, steps=1)["accuracy"]
print('Test accuract: {}'.format(test_acc * 100))


y_pred = dnn.predict(x=x_test, as_iterable=False)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
cnf_matrix = confusion_matrix(y_test, y_pred)
