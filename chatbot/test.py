import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
a = tf.sequence_mask([1, 3, 2], 5)
with tf.Session() as sess:
    
    print(sess.run(a))
