import tensorflow as tf
from tensorflow import keras
from config import D


def UPN(input_layer):
    x = keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same')(input_layer)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(filters=D.c_dim * D.scale * D.scale, kernel_size=(3, 3), padding='same')(x)
    x = tf.depth_to_space(x, block_size=D.scale)
    return x


def RDBs(input_layer):
    rdb_concat = []
    rdn_in = input_layer
    for i in range(1, D.D + 1):
        x = rdn_in
        for j in range(1, D.C + 1):
            tmp = keras.layers.Conv2D(filters=D.G, kernel_size=(D.kernel_size, D.kernel_size), padding='same')(x)
            tmp = keras.layers.ReLU()(tmp)
            x = tf.concat([x, tmp], axis=-1)
        x = keras.layers.Conv2D(filters=D.G, kernel_size=(1, 1), padding='same')(x)
        rdn_in = keras.layers.Add()([x, rdn_in])
        rdb_concat.append(rdn_in)
    return keras.layers.Concatenate()(rdb_concat)


def Weight_predict(input_layer):
    x = keras.layers.Dense(units=256, activation=tf.nn.relu)(input_layer)
    x = keras.layers.Dense(D.c_dim * D.meta_sr_c_dim * D.meta_sr_kernel_size * D.meta_sr_kernel_size)(x)
    return x


def batch_conv(inp, filters):
    """
        inp 的shape为[B, H, W, channels]
        filters 的shape为[B, kernel_size, kernel_size, channels, out_channels]
    """
    filters = tf.transpose(filters, perm=[1, 2, 0, 3, 4])
    filters_shape = tf.shape(filters)
    filters = tf.reshape(filters,
                         [filters_shape[0], filters_shape[1], filters_shape[2] * filters_shape[3], filters_shape[4]])
    inp_r = tf.transpose(inp, [1, 2, 0, 3])
    inp_shape = tf.shape(inp_r)
    inp_r = tf.reshape(inp_r, [1, inp_shape[0], inp_shape[1], inp_shape[2] * inp_shape[3]])
    padding = 'VALID'
    out = tf.nn.depthwise_conv2d(inp_r, filter=filters, strides=[1, 1, 1, 1], padding=padding)
    out = tf.reshape(out, [inp_shape[0] - filters_shape[0] + 1, inp_shape[1] - filters_shape[1] + 1, inp_shape[2], inp_shape[3], filters_shape[4]])
    out = tf.transpose(out, [2, 0, 1, 3, 4])
    out = tf.reduce_sum(out, axis=3)
    return out


def batch_conv_op(inp, filters):
    def single_conv(tupl):
        x, kernel = tupl
        return tf.nn.conv2d(x, kernel, strides=(1, 1, 1, 1), padding='VALID')

    batch_wise_conv = tf.squeeze(
        tf.map_fn(single_conv, (tf.expand_dims(inp, 1), filters), dtype=tf.float32, parallel_iterations=100,
                  swap_memory=True), axis=1)
    return batch_wise_conv


if __name__ == '__main__':
    # tf.enable_eager_execution()
    import time
    # import numpy as np
    sess = tf.Session()
    x = tf.random.normal(shape=[200000, 3, 3, 64])
    filters = tf.random.normal(shape=[200000, 3, 3, 64, 3])
    start = time.time()
    a = batch_conv(x, filters)
    sess.run(a)
    # b = batch_conv_op(x, filters)
    # sess.run(b)
    print(time.time() - start)


    # I = tf.ones(shape=[6, 128, 128, 27])
    # r = 3
    # print(_phase_shift(I, r))
