import cv2
from config import D
import tensorflow as tf
import glob
import numpy as np


def read_img(filename, scale=3):
    data = tf.io.read_file(filename)
    image = tf.image.decode_png(data, channels=3)
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    h = h // scale * scale
    w = w // scale * scale
    image = image[:h, :w, :]
    return image

def read_test_img(filename):
    data = tf.io.read_file(filename)
    image = tf.image.decode_png(data, channels=3)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)
    return image


def preprocess_img(image):
    h = np.shape(image)[0]
    w = np.shape(image)[1]
    probability_gaussian = np.random.uniform(0, 10, size=())
    if probability_gaussian > 5:
        temp_image = cv2.GaussianBlur(image, ksize=(7, 7), sigmaX=1.0)
    else:
        temp_image = image
    if D.model == 'RDN':
        scale = D.scale
    else:
        scale = D.scale
    input = cv2.resize(temp_image, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    h, w = np.shape(input)[:2]
    total_input = []
    total_label = []
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    for x in range(0, h * scale - D.image_size * scale + 1, D.stride * scale):
        for y in range(0, w * scale - D.image_size * scale + 1, D.stride * scale):
            sub_label = image[x:x + D.image_size * scale, y:y + D.image_size * scale]
            x_i = x // scale
            y_i = y // scale
            sub_input = input[x_i:x_i + D.image_size, y_i:y_i + D.image_size]
            noise = np.random.normal(size=np.shape(sub_input))
            probability_noise = np.random.uniform(0, 10, size=())
            probability_horizon = np.random.uniform(0, 10, size=())
            probability_top_down = np.random.uniform(0, 10, size=())
            probability_rorate = np.random.uniform(0, 10, size=())
            probability_ruihua = np.random.uniform(0, 10, size=())
            if probability_noise > 5:
                sub_input.astype(np.float32)
                sub_input = sub_input +  0.3 * noise
                sub_input = np.clip(sub_input, 0, 255)
                sub_input.astype(np.uint8)
            if probability_horizon > 5:
                sub_input = cv2.flip(sub_input, 1)
                sub_label = cv2.flip(sub_label, 1)
            if probability_top_down > 5:
                sub_input = cv2.flip(sub_input, 0)
                sub_label = cv2.flip(sub_label, 0)
            if probability_rorate > 5:
                sub_input = np.rot90(sub_input)
                sub_label = np.rot90(sub_label)
            if probability_ruihua > 5 and probability_gaussian > 5:
                sub_input = cv2.filter2D(sub_input, -1, kernel=kernel)
            sub_label = sub_label / 255.0
            sub_input = sub_input / 255.0
            sub_label = sub_label.astype(np.float32)
            sub_input = sub_input.astype(np.float32)
            total_input.append(sub_input)
            total_label.append(sub_label)
    return np.stack(total_input, axis=0).astype(np.float32), np.stack(total_label, axis=0).astype(np.float32)


def preprocess_metaSR_img(image):
    h = np.shape(image)[0]
    w = np.shape(image)[1]
    probability_gaussian = np.random.uniform(0, 10, size=())
    if probability_gaussian > 5:
        temp_image = cv2.GaussianBlur(image, ksize=(7, 7), sigmaX=1.0)
    else:
        temp_image = image
    if D.model == 'RDN':
        scale = D.scale
    else:
        scale = np.random.random_integers(2, 4, size=[])
    input = cv2.resize(temp_image, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    h, w = np.shape(input)[:2]
    total_input = []
    total_label = []
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    for x in range(0, h * scale - D.image_size * scale + 1, D.stride * scale):
        for y in range(0, w * scale - D.image_size * scale + 1, D.stride * scale):
            sub_label = image[x:x + D.image_size * scale, y:y + D.image_size * scale]
            x_i = x // scale
            y_i = y // scale
            sub_input = input[x_i:x_i + D.image_size, y_i:y_i + D.image_size]
            noise = np.random.normal(size=np.shape(sub_input))
            probability_noise = np.random.uniform(0, 10, size=())
            probability_horizon = np.random.uniform(0, 10, size=())
            probability_top_down = np.random.uniform(0, 10, size=())
            probability_rorate = np.random.uniform(0, 10, size=())
            probability_ruihua = np.random.uniform(0, 10, size=())
            if probability_noise > 5:
                sub_input.astype(np.float32)
                sub_input = sub_input +  0.3 * noise
                sub_input = np.clip(sub_input, 0, 255)
                sub_input.astype(np.uint8)
            if probability_horizon > 5:
                sub_input = cv2.flip(sub_input, 1)
                sub_label = cv2.flip(sub_label, 1)
            if probability_top_down > 5:
                sub_input = cv2.flip(sub_input, 0)
                sub_label = cv2.flip(sub_label, 0)
            if probability_rorate > 5:
                sub_input = np.rot90(sub_input)
                sub_label = np.rot90(sub_label)
            if probability_ruihua > 5 and probability_gaussian > 5:
                sub_input = cv2.filter2D(sub_input, -1, kernel=kernel)
            sub_label = sub_label / 255.0
            sub_input = sub_input / 255.0
            sub_label = sub_label.astype(np.float32)
            sub_input = sub_input.astype(np.float32)
            total_input.append(sub_input)
            total_label.append(sub_label)
    return np.stack(total_input, axis=0).astype(np.float32), np.stack(total_label, axis=0).astype(np.float32)


def preprocess_eval_image(image):
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    input = tf.image.resize_images(image, [h // 3, w // 3], tf.image.ResizeMethod.BICUBIC)
    input = tf.cast(input, tf.float32)
    image = tf.cast(image, tf.float32)
    # input = tf.divide(input, 255.0)
    # image = tf.divide(image, 255.0)
    input = tf.expand_dims(input, axis=0)
    image = tf.expand_dims(image, axis=0)
    return input, image

def preprocess_train_image(image):
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    input = tf.image.resize_images(image, [h // 3, w // 3], tf.image.ResizeMethod.BICUBIC)
    probability_noise = tf.random.uniform(shape=(), minval=0, maxval=10)
    probability_horizon = tf.random.uniform(shape=(), minval=0, maxval=10)
    probability_top_down = tf.random.uniform(shape=(), minval=0, maxval=10)
    probability_rorate = tf.random.uniform(shape=(), minval=0, maxval=10)
    input = tf.cond(probability_horizon > 5, lambda :tf.image.flip_left_right(input), lambda :input)
    image = tf.cond(probability_horizon > 5, lambda: tf.image.flip_left_right(image), lambda: image)
    input = tf.cond(probability_top_down > 5, lambda: tf.image.flip_up_down(input), lambda: input)
    image = tf.cond(probability_top_down > 5, lambda: tf.image.flip_up_down(image), lambda: image)
    input = tf.cond(probability_rorate > 5, lambda: tf.image.rot90(input), lambda: input)
    image = tf.cond(probability_rorate > 5, lambda: tf.image.rot90(image), lambda: image)
    input = tf.cast(input, tf.float32)
    image = tf.cast(image, tf.float32)
    noise = tf.random.normal(shape=tf.shape(input))
    input = tf.clip_by_value(tf.cond(probability_noise > 5, lambda :input + noise, lambda :input), clip_value_min=0, clip_value_max=255)
    input = tf.divide(input, 255.0)
    image = tf.divide(image, 255.0)
    input = tf.expand_dims(input, axis=0)
    image = tf.expand_dims(image, axis=0)
    return input, image

def generate_dict(features, labels, scales):
    return {"features":features, "scales":scales}, labels


def train_input_fn(filenames):
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(lambda x: read_img(x, 3))
    if D.model == 'RDN':
        dataset = dataset.map(lambda x:tf.py_func(preprocess_img, [x], Tout=[tf.float32, tf.float32]))
    else:
        dataset = dataset.map(lambda x:tf.py_func(preprocess_metaSR_img, [x], Tout=[tf.float32, tf.float32]))
    dataset = dataset.apply(tf.data.experimental.unbatch())
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(5000, 200))
    if D.model == 'RDN':
        dataset = dataset.batch(batch_size=D.batch_size)
    else:
        dataset = dataset.batch(batch_size=1)
    dataset = dataset.prefetch(-1)
    return dataset

def eval_input_fn(filenames):
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(lambda x: read_img(x, 3))
    dataset = dataset.map(preprocess_eval_image)
    dataset = dataset.prefetch(-1)
    return dataset

def train_input_fn_2(filenames):
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=100, count=200))
    dataset = dataset.map(lambda x: read_img(x, 3))
    dataset = dataset.map(preprocess_train_image)
    dataset = dataset.prefetch(-1)
    return dataset

# def train_metaSR_input_fn(filenames):
#     dataset = tf.data.Dataset.from_tensor_slices(filenames)
#     dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(128, 200))
#     dataset = dataset.map(lambda x: read_img(x, 3))
#     dataset = dataset.map(lambda x: tf.py_func(preprocess_metaSR_img, [x], Tout=[tf.float32, tf.float32]))
#     dataset = dataset.apply(tf.data.experimental.unbatch())
#     dataset = dataset.batch(batch_size=D.batch_size)
#     dataset = dataset.map(generate_dict)
#     dataset = dataset.prefetch(-1)
#     return dataset

def test_input_fn(filenames):
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(read_test_img)
    dataset = dataset.prefetch(-1)
    return dataset


if __name__ == '__main__':
    tf.enable_eager_execution()
    train_filenames = glob.glob('/home/admin-seu/sss/Dataset/DIV2K_train_HR/*')
    dataset = train_input_fn(train_filenames)
    for value in dataset:
        print(tf.shape(value[0]))
        print(tf.shape(value[1]))
