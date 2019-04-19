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
        features.set_shape([batch_size, image_size, image_size, 3])
        labels.set_shape([batch_size, image_size * 3, image_size * 3, 3])
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
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=tf.train.get_or_create_global_step())
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

    if mode == tf.estimator.ModeKeys.TRAIN:
        features.set_shape([batch_size, image_size, image_size, 3])
        labels.set_shape([batch_size, None, None, 3])
    F_1 = keras.layers.Conv2D(filters=G0, kernel_size=(ks, ks), padding='same')(features)
    F0 = keras.layers.Conv2D(filters=G, kernel_size=(ks, ks), padding='same')(F_1)
    FD = RDBs(F0)
    FGF1 = keras.layers.Conv2D(filters=G0, kernel_size=(1, 1), padding='same')(FD)
    FGF2 = keras.layers.Conv2D(filters=G0, kernel_size=(ks, ks), padding='same')(FGF1)
    FDF = keras.layers.Add()([FGF2, F_1])
    FU = UPN(FDF)
    IHR = keras.layers.Conv2D(filters=c_dim, kernel_size=(ks, ks), padding='same')(FU)
    label_shape = tf.shape(labels)
    feature_shape = tf.shape(features)
    scale = tf.cast(label_shape[1] / feature_shape[1], tf.float32)
    label_h = label_shape[1]
    label_w = label_shape[2]
    range_w = tf.range(0, label_w)
    range_h = tf.range(0, label_h)
    X, Y = tf.meshgrid(range_w, range_h)
    position = tf.stack([Y, X], axis=-1)
    position = tf.cast(position, tf.float32)
    position = tf.floordiv(position, scale)
    other = tf.ones(shape=[label_h, label_w, 1]) / scale
    position = tf.concat([position, other], axis=-1)
    position = tf.reshape(position, [-1, 3])
    output = Weight_predict(position)
    output = tf.reshape(output, [label_h, label_w, meta_sr_kernel_size, meta_sr_kernel_size, c_dim, meta_sr_c_dim])
    tensor_array = []
    IHR = tf.pad(IHR, [[0, 0], [1, 1], [1, 1], [0, 0]])
    if mode != tf.estimator.ModeKeys.PREDICT:
        for i in range(0, image_size * 3, 1):
            for j in range(0, image_size * 3, 1):
                feature = IHR[:, i//3:i//3+meta_sr_kernel_size, j//3:j//3+meta_sr_kernel_size,:]
                temp = output[i, j]
                pixel = tf.nn.convolution(feature, temp, padding='VALID', strides=[2, 2])
                tensor_array.append(pixel)
        arrays = tf.stack(tensor_array, axis=0)
        arrays = tf.reshape(arrays, [label_h, label_w, -1, meta_sr_c_dim])
        arrays = tf.transpose(arrays, perm=[2, 0, 1, 3])
        loss = tf.reduce_mean(tf.abs(labels - arrays))
        tf.summary.scalar('loss', loss)
        tf.summary.image('SR_image', IHR, max_outputs=3)
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=tf.train.get_or_create_global_step(), var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        else:
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss)
    else:
        predictions = {"image": IHR}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


if __name__ == '__main__':
    num_gpus = 1
    model_dir = "./models_metaSR_v1"
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
                                        log_step_count_steps=100, save_checkpoints_steps=2000,
                                        eval_distribute=strategy, save_summary_steps=500)
        estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, config=config,
                                           params=params)
        # estimator = tf.estimator.Estimator(model_fn=model_fn_meta_SR, model_dir=model_dir, config=config,
        #                                    params=params)
    else:
        config = tf.estimator.RunConfig(save_checkpoints_steps=1000, log_step_count_steps=10)
        estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=params,
                                           config=config)
        # estimator = tf.estimator.Estimator(model_fn=model_fn_meta_SR, model_dir=model_dir, config=config,
        #                                    params=params)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda :train_input_fn(train_filenames), max_steps=200000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda :train_input_fn(test_filenames), throttle_secs=100)
    # res = estimator.predict(lambda :test_input_fn(eval_filenames))
    # print(eval_filenames)
    # for idx, ele in enumerate(res):
    #     image = ele['image']
    #     image = image * 255
    #     image = np.clip(image, 0, 255)
    #     image = image.astype(np.uint8)
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite('./result/{}'.format(os.path.basename(eval_filenames[idx])), image)
    #     if idx == 100:
    #         break
    tf.estimator.train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)