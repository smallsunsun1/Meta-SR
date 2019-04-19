import tensorflow as tf
from tensorflow import keras
from config import D


def _phase_shift(I, r):
    # Helper function with main phase shift operation
    shape = tf.shape(I)
    bsize = shape[0]
    a = shape[1]
    b = shape[2]
    c = shape[3]
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.unstack(X, axis=1)   # a, [bsize, b, r, r]
    # X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
    X = tf.unstack(X, axis=1)  # b, [bsize, a*r, r]
    # X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a * r, b * r, 1))


def PS(X, r):
    # Main Op that you can arbitrary use in your tensorflow code
    Xc = tf.split(X, 3, axis=3)
    X = tf.concat([_phase_shift(x, r) for x in Xc], axis=3)
    return X


def UPN(input_layer):
    x = keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same')(input_layer)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(filters=D.c_dim * D.scale * D.scale, kernel_size=(3, 3), padding='same')(x)
    # x = PS(x, D.scale)
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


if __name__ == '__main__':
    tf.enable_eager_execution()
    I = tf.ones(shape=[6, 128, 128, 27])
    r = 3
    print(_phase_shift(I, r))
