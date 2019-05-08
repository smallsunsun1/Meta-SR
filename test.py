import tensorflow as tf
import networkx as nx
from matplotlib import pyplot as plt

upsample_scale = 3
meta_sr_kernel_size = 3
a = tf.random.normal(shape=[60*60*6*3*3*3])
# a = tf.range(60*60*6*3*3*3)
filters = tf.reshape(a, [60, 60, 3, 3, 6, 3])
feature = tf.ones(shape=[10, 20, 20, 6])
feature = tf.pad(feature, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
b = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)

def cond(i, j, h, w, b):
    return tf.less(i, h)

def body(i, j, h, w, b):
    index = i * w + j
    temp = feature[:, i // upsample_scale:i // upsample_scale + meta_sr_kernel_size,
              j // upsample_scale:j // upsample_scale + meta_sr_kernel_size, :]
    pixel = tf.nn.conv2d(temp, filters[i, j], strides=[1, 1, 1, 1], padding='VALID')
    pixel = tf.squeeze(pixel)
    b = b.write(index, pixel)
    i= tf.cond(tf.equal(j + 1, w), lambda :i+1, lambda :i)
    j = tf.mod(j + 1, w)
    return i, j, h, w, b

_ , _, _, _, res = tf.while_loop(cond, body, [0, 0, 20*upsample_scale, 20*upsample_scale, b])
sess = tf.Session()
print(sess.run(tf.shape(res.stack())))