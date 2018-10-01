import tensorflow as tf
import numpy as np
import IPython

batch_size = 256

# Load mnist data.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Model.
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
labels = tf.placeholder(dtype=tf.int32, shape=[None])
one_hot_labels = tf.one_hot(labels, batch_size)

net_1 = tf.layers.flatten(x)
net_2 = tf.layers.dense(net_1, 100, activation=tf.nn.relu)
net_3 = tf.layers.dense(net_2, 100, activation=tf.nn.relu)

# Need to chain these together. tf.unstack breaks apart the batch, but the way
# tf does unrolling basically means you can only get recurrent connections w/in
# each batch? need to investigate further.

logits = tf.layers.dense(net, 10, activation=None)
preds = tf.argmax(logits, axis=1)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

train = tf.train.AdamOptimizer(0.0001).minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for i in range(10000):
    sample = np.random.choice(xrange(x_train.shape[0]), size=batch_size, replace=False)
    x_sample = x_train[sample]
    y_sample = y_train[sample]
    sess.run([loss, train], {x: x_sample, labels: y_sample})

    if i % 100 == 0:
      sample = np.random.choice(xrange(x_test.shape[0]), size=1000, replace=False)
      x_sample_test = x_test[sample]
      y_sample_test = y_test[sample]
      predictions = sess.run(preds, {x: x_sample_test})
      print 'Accuracy: %f' % (1.0*np.sum(predictions == y_sample_test) / 1000)


  IPython.embed()
