import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# set CSV file list
filename_queue = tf.train.string_input_producer(
    ["data-iris-1.csv", "data-iris-2.csv"], shuffle=False, name='filename_queue')

# set tensorflow reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# set record_defaults corresponding to data form
record_defaults = [[0.]]*5  # record_defaults = [[0.], [0.], [0.], [0.], [0.]]
data = tf.decode_csv(value, record_defaults=record_defaults)

# set collecting data and batch option
train_x_batch, train_y_batch = tf.train.batch([data[0:-1], data[-1:]], batch_size=4)

sess = tf.Session()

# start (mandatory)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    print(x_batch.shape, y_batch.shape) # print shape of each variable
    print(x_batch, y_batch)             # print data of each variable

# end (mandatory)
coord.request_stop()
coord.join(threads)
