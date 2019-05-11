from config import D
import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib import distribute
from basemodel import *
from utils import *
import glob
import cv2
import os
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)


def model_fn(features, labels, mode, params):
    G0 = params.get('G0', 64)
    G = params.get('G', 64)
    c_dim = params.get('c_dim', 3)
    ks = params.get('kernel_size', 3)
    batch_size = params.get('batch_size', 32)
    learning_rate = params.get('learning_rate', 0.0001)
    image_size = params.get('image_size', 32)
    if mode == tf.estimator.ModeKeys.TRAIN:
        features.set_shape([None, None, None, 3])
        labels.set_shape([None, None, None, 3])
    F_1 = keras.layers.Conv2D(filters=G0, kernel_size=(ks, ks), padding='same')(features)
    F0 = keras.layers.Conv2D(filters=G, kernel_size=(ks, ks), padding='same')(F_1)
    FD = RDBs(F0)
    FGF1 = keras.layers.Conv2D(filters=G0, kernel_size=(1, 1), padding='same')(FD)
    FGF2 = keras.layers.Conv2D(filters=G0, kernel_size=(ks, ks), padding='same')(FGF1)
    FDF = keras.layers.Add()([FGF2, F_1])
    FU = UPN(FDF)
    IHR = keras.layers.Conv2D(filters=c_dim, kernel_size=(ks, ks), padding='same')(FU)
    if mode != tf.estimator.ModeKeys.PREDICT:
        loss = tf.reduce_mean(tf.abs(labels - IHR))
        tf.summary.scalar('loss', loss)
        tf.summary.image('SR_image', IHR, max_outputs=3)
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss,
                                                                      global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        else:
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss)
    else:
        predictions = {"image": IHR}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


def model_fn_meta_SR(features, labels, mode, params):
    G0 = params.get('G0', 64)
    G = params.get('G', 64)
    c_dim = params.get('c_dim', 3)
    Dscale = params.get('scale', 3)
    ks = params.get('kernel_size', 3)
    batch_size = params.get('batch_size', 32)
    learning_rate = params.get('learning_rate', 0.0001)
    image_size = params.get('image_size', 32)
    meta_sr_c_dim = params.get('meta_sr_c_dim', 3)
    meta_sr_kernel_size = params.get('meta_sr_kernel_size', 3)
    meta_sr_upsample_scale = params.get('meta_sr_upsample_scale', 3)

    if mode == tf.estimator.ModeKeys.TRAIN:
        features.set_shape([batch_size, image_size, image_size, 3])
        labels.set_shape([batch_size, None, None, 3])
    if mode == tf.estimator.ModeKeys.EVAL:
        features.set_shape([batch_size, None, None, 3])
        labels.set_shape([batch_size, None, None, 3])
    F_1 = keras.layers.Conv2D(filters=G0, kernel_size=(ks, ks), padding='same')(features)
    F0 = keras.layers.Conv2D(filters=G, kernel_size=(ks, ks), padding='same')(F_1)
    FD = RDBs(F0)
    FGF1 = keras.layers.Conv2D(filters=G0, kernel_size=(1, 1), padding='same')(FD)
    FGF2 = keras.layers.Conv2D(filters=G0, kernel_size=(ks, ks), padding='same')(FGF1)
    FDF = keras.layers.Add()([FGF2, F_1])
    # FU = UPN(FDF)
    IHR = keras.layers.Conv2D(filters=c_dim, kernel_size=(ks, ks), padding='same')(FDF)
    feature_shape = tf.shape(features)
    if mode != tf.estimator.ModeKeys.PREDICT:
        label_shape = tf.shape(labels)
        scale = tf.cast(label_shape[1] / feature_shape[1], tf.float32)
        label_h = label_shape[1]
        label_w = label_shape[2]
        ratio_w = tf.divide(label_w, feature_shape[2])
        ratio_h = tf.divide(label_h, feature_shape[1])
        range_w = tf.range(0, ratio_w)
        range_h = tf.range(0, ratio_h)
    else:
        range_w = tf.range(0, meta_sr_upsample_scale)
        range_h = tf.range(0, meta_sr_upsample_scale)
        scale = float(meta_sr_upsample_scale)
        ratio_h = meta_sr_upsample_scale
        ratio_w = meta_sr_upsample_scale
    X, Y = tf.meshgrid(range_w, range_h)
    position = tf.stack([Y, X], axis=-1)
    position = tf.cast(position, tf.float32)
    position = position - tf.floordiv(position, scale)
    other = tf.ones(shape=[ratio_h, ratio_w, 1]) / scale
    position = tf.concat([position, other], axis=-1)
    position = tf.reshape(position, [-1, 3])
    output = Weight_predict(position)
    output = tf.reshape(output, [ratio_h, ratio_w, meta_sr_kernel_size, meta_sr_kernel_size, c_dim, meta_sr_c_dim])
    """注释掉用于真正的meta-sr模式"""
    # output = tf.reduce_mean(output, axis=[0, 1])
    tensor_array = []
    IHR = tf.pad(IHR, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    if mode != tf.estimator.ModeKeys.PREDICT:
        for i in range(0, image_size * Dscale, 1):
            for j in range(0, image_size * Dscale, 1):
                feature = IHR[:, i // Dscale:i // Dscale + meta_sr_kernel_size,
                          j // Dscale:j // Dscale + meta_sr_kernel_size, :]

                """This part of code is true implementation, Not test yet"""
                pixel = tf.nn.conv2d(feature, output[i % Dscale, j % Dscale], strides=[1, 1, 1, 1], padding='VALID')
                pixel = tf.squeeze(pixel)
                tensor_array.append(pixel)

                """This part is new simple implementation, Is training"""
                # tensor_array.append(feature)

        arrays = tf.stack(tensor_array, axis=0)
        """This part of code is true implementation, Not test yet"""
        arrays = tf.reshape(arrays, [label_h, label_w, -1, 3])
        result = tf.transpose(arrays, perm=[2, 0, 1, 3])

        """This part is new simple implementation, Is training"""
        # result= tf.reshape(arrays, [label_h * Dscale, label_w * Dscale, -1, c_dim])
        # result = tf.transpose(result, perm=[2, 0, 1, 3])
        # result = tf.nn.conv2d(result, filter=output, strides=[1, Dscale, Dscale, 1], padding='VALID')

        print("here is the shape: ", result.get_shape().as_list())
        loss = tf.reduce_mean(tf.abs(labels - result))
        tf.summary.scalar('loss', loss)
        tf.summary.image('SR_image', result, max_outputs=10)
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss,
                                                                      global_step=tf.train.get_or_create_global_step(),
                                                                      var_list=tf.get_collection(
                                                                          tf.GraphKeys.TRAINABLE_VARIABLES))
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        else:
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss)
    else:
        output_shape = feature_shape * meta_sr_upsample_scale
        b = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
        def cond(i, j, h, w, b):
            return tf.less(i, h)

        def body(i, j, h, w, b):
            index = i * w + j
            temp = IHR[:, i // meta_sr_upsample_scale:i // meta_sr_upsample_scale + meta_sr_kernel_size,
                   j // meta_sr_upsample_scale:j // meta_sr_upsample_scale + meta_sr_kernel_size, :]
            pixel = tf.nn.conv2d(temp, output[tf.mod(i, meta_sr_upsample_scale) , tf.mod(j, meta_sr_upsample_scale)], strides=[1, 1, 1, 1], padding='VALID')
            pixel = tf.squeeze(pixel)
            b = b.write(index, pixel)
            i = tf.cond(tf.equal(j + 1, w), lambda: i + 1, lambda: i)
            j = tf.mod(j + 1, w)
            return i, j, h, w, b

        _, _, _, _, res = tf.while_loop(cond, body, [0, 0, feature_shape[1] * meta_sr_upsample_scale, feature_shape[2] * meta_sr_upsample_scale, b])
        res = res.stack()
        res = tf.reshape(res, shape=[output_shape[1], output_shape[2], -1, 3])
        res = tf.transpose(res, perm=[2, 0, 1, 3])
        predictions = {"image": res}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


def model_fn_meta_SR_new(features, labels, mode, params):
    G0 = params.get('G0', 64)
    G = params.get('G', 64)
    c_dim = params.get('c_dim', 3)
    Dscale = params.get('scale', 3)
    ks = params.get('kernel_size', 3)
    batch_size = params.get('batch_size', 32)
    learning_rate = params.get('learning_rate', 0.0001)
    image_size = params.get('image_size', 32)
    meta_sr_c_dim = params.get('meta_sr_c_dim', 3)
    meta_sr_kernel_size = params.get('meta_sr_kernel_size', 3)
    meta_sr_upsample_scale = params.get('meta_sr_upsample_scale', 3)

    if mode == tf.estimator.ModeKeys.TRAIN:
        features.set_shape([batch_size, None, None, 3])
        labels.set_shape([batch_size, None, None, 3])
    if mode == tf.estimator.ModeKeys.EVAL:
        features.set_shape([batch_size, None, None, 3])
        labels.set_shape([batch_size, None, None, 3])
    F_1 = keras.layers.Conv2D(filters=G0, kernel_size=(ks, ks), padding='same')(features)
    F0 = keras.layers.Conv2D(filters=G, kernel_size=(ks, ks), padding='same')(F_1)
    FD = RDBs(F0)
    FGF1 = keras.layers.Conv2D(filters=G0, kernel_size=(1, 1), padding='same')(FD)
    FGF2 = keras.layers.Conv2D(filters=G0, kernel_size=(ks, ks), padding='same')(FGF1)
    FDF = keras.layers.Add()([FGF2, F_1])
    # FU = UPN(FDF)
    IHR = keras.layers.Conv2D(filters=c_dim, kernel_size=(ks, ks), padding='same')(FDF)
    feature_shape = tf.shape(features)
    if mode != tf.estimator.ModeKeys.PREDICT:
        label_shape = tf.shape(labels)
        scale = tf.cast(label_shape[1] / feature_shape[1], tf.float32)
        label_h = label_shape[1]
        label_w = label_shape[2]
    else:
        scale = float(meta_sr_upsample_scale)
    if mode == tf.estimator.ModeKeys.PREDICT:
        output_shape = feature_shape * meta_sr_upsample_scale
        Scale = meta_sr_upsample_scale
    else:
        output_shape = feature_shape * Dscale
        Scale = Dscale
    X, Y = tf.meshgrid(tf.range(output_shape[2]), tf.range(output_shape[1]))
    position = tf.stack([Y, X], axis=-1)
    position = tf.cast(position, tf.float32)
    position = tf.divide(position, scale) - tf.floordiv(position, scale)
    other = tf.ones(shape=[output_shape[1], output_shape[2], 1]) / scale
    position = tf.concat([position, other], axis=-1)
    position = tf.reshape(position, [-1, 3])
    output = Weight_predict(position)
    output = tf.reshape(output, [output_shape[1], output_shape[2], meta_sr_kernel_size, meta_sr_kernel_size, c_dim, meta_sr_c_dim])
    """注释掉用于真正的meta-sr模式"""
    IHR = tf.pad(IHR, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    b = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
    def cond(i, j, h, w, b):
        return tf.less(i, h)

    def body(i, j, h, w, b):
        index = i * w + j
        temp = IHR[:, i // Scale:i // Scale + meta_sr_kernel_size,
               j // Scale:j // Scale + meta_sr_kernel_size, :]
        pixel = tf.nn.conv2d(temp, output[i , j], strides=[1, 1, 1, 1], padding='VALID')
        pixel = tf.squeeze(pixel)
        b = b.write(index, pixel)
        i = tf.cond(tf.equal(j + 1, w), lambda: i + 1, lambda: i)
        j = tf.mod(j + 1, w)
        return i, j, h, w, b

    _, _, _, _, res = tf.while_loop(cond, body, [0, 0, output_shape[1], output_shape[2], b])
    res = res.stack()
    res = tf.reshape(res, shape=[output_shape[1], output_shape[2], -1, 3])
    res = tf.transpose(res, perm=[2, 0, 1, 3])
    predictions = {"image": res}
    if mode != tf.estimator.ModeKeys.PREDICT:
        loss = tf.reduce_mean(tf.abs(labels - res))
        tf.summary.scalar('loss', loss)
        tf.summary.image('image', res, max_outputs=10)
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss,
                                                                      global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss)
    else:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

# def model_fn_meta_SR_new_2(features, labels, mode, params):
#     G0 = params.get('G0', 64)
#     G = params.get('G', 64)
#     c_dim = params.get('c_dim', 3)
#     Dscale = params.get('scale', 3)
#     ks = params.get('kernel_size', 3)
#     batch_size = params.get('batch_size', 32)
#     learning_rate = params.get('learning_rate', 0.0001)
#     image_size = params.get('image_size', 32)
#     meta_sr_c_dim = params.get('meta_sr_c_dim', 3)
#     meta_sr_kernel_size = params.get('meta_sr_kernel_size', 3)
#     meta_sr_upsample_scale = params.get('meta_sr_upsample_scale', 3)
#
#     if mode == tf.estimator.ModeKeys.TRAIN:
#         features.set_shape([batch_size, image_size, image_size, 3])
#         labels.set_shape([batch_size, None, None, 3])
#     if mode == tf.estimator.ModeKeys.EVAL:
#         features.set_shape([batch_size, None, None, 3])
#         labels.set_shape([batch_size, None, None, 3])
#     F_1 = keras.layers.Conv2D(filters=G0, kernel_size=(ks, ks), padding='same')(features)
#     F0 = keras.layers.Conv2D(filters=G, kernel_size=(ks, ks), padding='same')(F_1)
#     FD = RDBs(F0)
#     FGF1 = keras.layers.Conv2D(filters=G0, kernel_size=(1, 1), padding='same')(FD)
#     FGF2 = keras.layers.Conv2D(filters=G0, kernel_size=(ks, ks), padding='same')(FGF1)
#     FDF = keras.layers.Add()([FGF2, F_1])
#     # FU = UPN(FDF)
#     IHR = keras.layers.Conv2D(filters=c_dim, kernel_size=(ks, ks), padding='same')(FDF)
#     feature_shape = tf.shape(features)
#     if mode != tf.estimator.ModeKeys.PREDICT:
#         label_shape = tf.shape(labels)
#         scale = tf.cast(label_shape[1] / feature_shape[1], tf.float32)
#         label_h = label_shape[1]
#         label_w = label_shape[2]
#         ratio_w = tf.divide(label_w, feature_shape[2])
#         ratio_h = tf.divide(label_h, feature_shape[1])
#         range_w = tf.range(0, ratio_w)
#         range_h = tf.range(0, ratio_h)
#     else:
#         range_w = tf.range(0, meta_sr_upsample_scale)
#         range_h = tf.range(0, meta_sr_upsample_scale)
#         scale = float(meta_sr_upsample_scale)
#         ratio_h = meta_sr_upsample_scale
#         ratio_w = meta_sr_upsample_scale
#     X, Y = tf.meshgrid(range_w, range_h)
#     position = tf.stack([Y, X], axis=-1)
#     position = tf.cast(position, tf.float32)
#     position = position - tf.floordiv(position, scale)
#     other = tf.ones(shape=[ratio_h, ratio_w, 1]) / scale
#     position = tf.concat([position, other], axis=-1)
#     position = tf.reshape(position, [-1, 3])
#     output = Weight_predict(position)
#     output = tf.reshape(output, [ratio_h, ratio_w, meta_sr_kernel_size, meta_sr_kernel_size, c_dim, meta_sr_c_dim])
#     """注释掉用于真正的meta-sr模式"""
#
#     if mode == tf.estimator.ModeKeys.PREDICT:
#         output_shape = feature_shape * meta_sr_upsample_scale
#         Scale = meta_sr_upsample_scale
#     else:
#         output_shape = feature_shape * Dscale
#         Scale = Dscale
#     IHR = tf.tile(IHR, multiples=[1, Scale, Scale, 1])
#     IHR = tf.space_to_batch_nd(IHR, block_shape=[1, Scale], paddings=[[0, 0],
#                                                               [0, 0]])
#     IHR = tf.reshape(IHR, shape=[feature_shape[0], feature_shape[1]*3, feature_shape[2]*3, 3])
#     IHR = tf.pad(IHR, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
#
#     predictions = {"image": res}
#     if mode != tf.estimator.ModeKeys.PREDICT:
#         loss = tf.reduce_mean(tf.abs(labels - res))
#         tf.summary.scalar('loss', loss)
#         tf.summary.image('image', res, max_outputs=10)
#         if mode == tf.estimator.ModeKeys.TRAIN:
#             train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss,
#                                                                       global_step=tf.train.get_or_create_global_step())
#             return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
#         elif mode == tf.estimator.ModeKeys.EVAL:
#             return tf.estimator.EstimatorSpec(mode=mode, loss=loss)
#     else:
#         return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


if __name__ == '__main__':
    num_gpus = 2
    model_dir = "./models_metaSR_new_v2"
    # model_dir = './models_v3'
    train_filenames = glob.glob('/home/admin-seu/sss/Dataset/DIV2K_train_HR/*')
    eval_filenames = glob.glob('/home/admin-seu/sss/Dataset/test_data/*')
    test_filenames = glob.glob('/home/admin-seu/sss/Dataset/DIV2K_test_HR/*')
    params = dict(D)
    print(params)
    if num_gpus > 0:
        strategy = distribute.MirroredStrategy(num_gpus=num_gpus)
        session_configs = tf.ConfigProto(allow_soft_placement=True)
        session_configs.gpu_options.allow_growth = True
        config = tf.estimator.RunConfig(train_distribute=strategy, session_config=session_configs,
                                        log_step_count_steps=50, save_checkpoints_steps=2000,
                                        eval_distribute=strategy, save_summary_steps=500)
        if D.model == 'RDN':
            Estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, config=config,
                                               params=params)
        else:
            Estimator = tf.estimator.Estimator(model_fn=model_fn_meta_SR_new, model_dir=model_dir, config=config,
                                           params=params)
    else:
        config = tf.estimator.RunConfig(save_checkpoints_steps=1000, log_step_count_steps=10)
        if D.model == 'RDN':
            Estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=params,
                                               config=config)
        else:
            Estimator = tf.estimator.Estimator(model_fn=model_fn_meta_SR_new, model_dir=model_dir, config=config,
                                           params=params)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(train_filenames), max_steps=200000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: train_input_fn(test_filenames), throttle_secs=100, steps=500)
    if D.mode == 'predict':
        res = Estimator.predict(lambda :test_input_fn(eval_filenames))
        print(eval_filenames)
        for idx, ele in enumerate(res):
            print("proprocess image {}".format(idx))
            image = ele['image']
            image = image * 255
            image = np.clip(image, 0, 255)
            image = image.astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite('./result_v2/{}'.format(os.path.basename(eval_filenames[idx])), image)
            if idx == 100:
                break
    else:
        tf.estimator.train_and_evaluate(Estimator, train_spec=train_spec, eval_spec=eval_spec)
