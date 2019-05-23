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
                sub_input = sub_input + 0.3 * noise
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
    indices = np.arange(0, len(total_input))
    indices = np.random.shuffle(indices)
    indices = indices[:D.batch_size]
    output1 = np.stack(total_input, axis=0).astype(np.float32)[indices]
    output2 = np.stack(total_label, axis=0).astype(np.float32)[indices]
    return output1[indices], output2[indices]


def preprocess_metaSR_img(image):
    if D.model == 'RDN':
        scale = D.scale
    else:
        scale = np.random.random_integers(2, 4, size=[])
    h = np.shape(image)[0]
    w = np.shape(image)[1]
    h_ = np.shape(image)[0]
    w_ = np.shape(image)[1]
    h_ = h_ // scale * scale
    w_ = w_ // scale * scale
    image = image[:h_, :w_, :]
    probability_gaussian = np.random.uniform(0, 10, size=())
    if probability_gaussian > 5:
        temp_image = cv2.GaussianBlur(image, ksize=(7, 7), sigmaX=1.0)
    else:
        temp_image = image
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
                sub_input = sub_input + 0.3 * noise
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
    indices = np.arange(0, len(total_input))
    np.random.shuffle(indices)
    indices = indices[:D.batch_size]
    output1 = np.stack(total_input, axis=0).astype(np.float32)
    output2 = np.stack(total_label, axis=0).astype(np.float32)
    return output1[indices], output2[indices]


def tf_preprocess_metaSR_img(image):
    scale = tf.random.uniform(shape=(), minval=2, maxval=5, dtype=tf.int32)
    h_ = tf.shape(image)[0]
    w_ = tf.shape(image)[1]
    h_ = h_ // scale * scale
    w_ = w_ // scale * scale
    image = image[:h_, :w_, :]
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    image = tf.cast(image, tf.float32)
    probability_gaussian = tf.random.uniform(shape=(), minval=0, maxval=10)
    temp_image = tf.cond(tf.greater(probability_gaussian, 5), lambda: image,
                         lambda: image + tf.random.truncated_normal(shape=tf.shape(image)))
    temp_image = tf.clip_by_value(temp_image, 0, 255)
    if D.model == 'RDN':
        scale = D.scale
    input = tf.squeeze(tf.image.resize_bicubic(tf.expand_dims(temp_image, 0), (h // scale, w // scale)))
    h_ = tf.shape(input)[0]
    w_ = tf.shape(input)[1]
    total_input = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
    total_label = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
    stride = tf.convert_to_tensor(D.stride, tf.int32)
    image_size = tf.convert_to_tensor(D.image_size, tf.int32)

    def cond(i, j, h, w, b1, b2, index):
        return tf.less(i, h)

    def body(i, j, h, w, b1, b2, index):
        sub_label = image[i * scale: i * scale + image_size * scale, j * scale:j * scale + image_size * scale]
        sub_input = input[i:i + image_size, j:j + image_size]
        probability_horizon = tf.random.uniform(shape=[], maxval=10)
        probability_top_down = tf.random.uniform(shape=[], maxval=10)
        probability_rotate = tf.random.uniform(shape=[], maxval=10)
        sub_input = tf.cond(probability_horizon > 5, lambda: tf.image.flip_left_right(sub_input), lambda: sub_input)
        sub_label = tf.cond(probability_horizon > 5, lambda: tf.image.flip_left_right(sub_label), lambda: sub_label)
        sub_input = tf.cond(probability_top_down > 5, lambda: tf.image.flip_up_down(sub_input), lambda: sub_input)
        sub_label = tf.cond(probability_top_down > 5, lambda: tf.image.flip_up_down(sub_label), lambda: sub_label)
        sub_input = tf.cond(probability_rotate > 5, lambda: tf.image.rot90(sub_input), lambda: sub_input)
        sub_label = tf.cond(probability_rotate > 5, lambda: tf.image.rot90(sub_label), lambda: sub_label)
        sub_input = tf.cast(sub_input / 255.0, tf.float32)
        sub_label = tf.cast(sub_label / 255.0, tf.float32)
        b1 = b1.write(index, sub_input)
        b2 = b2.write(index, sub_label)
        i = tf.cond(tf.greater_equal(j + stride, w), lambda: i + stride, lambda: i)
        j = tf.cond(tf.greater_equal(j + stride, w), lambda: 0, lambda: j + stride)
        index = tf.add(index, 1)
        return i, j, h, w, b1, b2, index

    i_, j_, h, w, arr1, arr2, _ = tf.while_loop(cond, body,
                                                [0, 0, h_ - image_size, w_ - image_size, total_input, total_label, 0])
    arr1 = arr1.stack()
    arr2 = arr2.stack()
    return arr1, arr2

def tf_preprocess_metaSR_img_new(image):
    scale = tf.random.uniform(shape=(), minval=2, maxval=5, dtype=tf.int32)
    h_ = tf.shape(image)[0]
    w_ = tf.shape(image)[1]
    h_ = h_ // scale * scale
    w_ = w_ // scale * scale
    image = image[:h_, :w_, :]
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    image = tf.cast(image, tf.float32)
    probability_gaussian = tf.random.uniform(shape=(), minval=0, maxval=10)
    temp_image = tf.cond(tf.greater(probability_gaussian, 5), lambda: image,
                         lambda: image + tf.random.truncated_normal(shape=tf.shape(image)))
    temp_image = tf.clip_by_value(temp_image, 0, 255)
    image_size = D.image_size
    input = tf.squeeze(tf.image.resize_bicubic(tf.expand_dims(temp_image, 0), (h // scale, w // scale)))
    h_i = tf.shape(input)[0]
    w_i = tf.shape(input)[1]
    x_range = tf.range(h_i - image_size)
    y_range = tf.range(w_i - image_size)
    x_range = tf.random.shuffle(x_range)
    y_range = tf.random.shuffle(y_range)
    size = 8
    x_indices = x_range[:size]
    y_indices = y_range[:size]
    total_input = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
    total_label = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
    xy_indices = tf.meshgrid(x_indices, y_indices)
    xy_indices = tf.stack(xy_indices, axis=-1)
    xy_indices = tf.reshape(xy_indices, [-1, 2])
    def cond(i, n, b1, b2):
        return i < n
    def body(i, n, b1, b2):
        sub_label = image[xy_indices[i][0] * scale: xy_indices[i][0] * scale + image_size * scale, xy_indices[i][1] * scale:xy_indices[i][1] * scale + image_size * scale]
        sub_input = input[xy_indices[i][0]: xy_indices[i][0] + image_size, xy_indices[i][1]:xy_indices[i][1] + image_size]
        probability_horizon = tf.random.uniform(shape=[], maxval=10)
        probability_top_down = tf.random.uniform(shape=[], maxval=10)
        probability_rotate = tf.random.uniform(shape=[], maxval=10)
        sub_input = tf.cond(probability_horizon > 5, lambda: tf.image.flip_left_right(sub_input), lambda: sub_input)
        sub_label = tf.cond(probability_horizon > 5, lambda: tf.image.flip_left_right(sub_label), lambda: sub_label)
        sub_input = tf.cond(probability_top_down > 5, lambda: tf.image.flip_up_down(sub_input), lambda: sub_input)
        sub_label = tf.cond(probability_top_down > 5, lambda: tf.image.flip_up_down(sub_label), lambda: sub_label)
        sub_input = tf.cond(probability_rotate > 5, lambda: tf.image.rot90(sub_input), lambda: sub_input)
        sub_label = tf.cond(probability_rotate > 5, lambda: tf.image.rot90(sub_label), lambda: sub_label)
        sub_input = tf.cast(sub_input / 255.0, tf.float32)
        sub_label = tf.cast(sub_label / 255.0, tf.float32)
        b1 = b1.write(i, sub_input)
        b2 = b2.write(i, sub_label)
        i = tf.add(i, 1)
        return i, n, b1, b2
    _, _, arr1, arr2 = tf.while_loop(cond, body, [0, size * size, total_input, total_label])
    arr1 = arr1.stack()
    arr2 = arr2.stack()
    return arr1, arr2



def preprocess_eval_image(image):
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    if D.model == 'RDN':
        input = tf.image.resize_images(image, [h // D.scale, w // D.scale], tf.image.ResizeMethod.BICUBIC)
    else:
        input = tf.image.resize_images(image, [h // D.meta_sr_upsample_scale, w // D.meta_sr_upsample_scale], tf.image.ResizeMethod.BICUBIC)
    input = tf.cast(input, tf.float32)
    image = tf.cast(image, tf.float32)
    input = tf.divide(input, 255.0)
    image = tf.divide(image, 255.0)
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
    input = tf.cond(probability_horizon > 5, lambda: tf.image.flip_left_right(input), lambda: input)
    image = tf.cond(probability_horizon > 5, lambda: tf.image.flip_left_right(image), lambda: image)
    input = tf.cond(probability_top_down > 5, lambda: tf.image.flip_up_down(input), lambda: input)
    image = tf.cond(probability_top_down > 5, lambda: tf.image.flip_up_down(image), lambda: image)
    input = tf.cond(probability_rorate > 5, lambda: tf.image.rot90(input), lambda: input)
    image = tf.cond(probability_rorate > 5, lambda: tf.image.rot90(image), lambda: image)
    input = tf.cast(input, tf.float32)
    image = tf.cast(image, tf.float32)
    noise = tf.random.normal(shape=tf.shape(input))
    input = tf.clip_by_value(tf.cond(probability_noise > 5, lambda: input + noise, lambda: input), clip_value_min=0,
                             clip_value_max=255)
    input = tf.divide(input, 255.0)
    image = tf.divide(image, 255.0)
    input = tf.expand_dims(input, axis=0)
    image = tf.expand_dims(image, axis=0)
    return input, image


def generate_dict(features, labels, scales):
    return {"features": features, "scales": scales}, labels


def train_input_fn(filenames):
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(5000, 200))
    dataset = dataset.map(lambda x: read_img(x, 1))
    if D.model == 'RDN':
        dataset = dataset.map(lambda x: tf.py_func(preprocess_img, [x], Tout=[tf.float32, tf.float32]))
    else:
        dataset = dataset.map(lambda x: tf.py_func(preprocess_metaSR_img, [x], Tout=[tf.float32, tf.float32]))
        # dataset = dataset.map(tf_preprocess_metaSR_img, num_parallel_calls=20)
    # dataset = dataset.apply(tf.data.experimental.unbatch())
    # dataset = dataset.batch(batch_size=D.batch_size)
    dataset = dataset.prefetch(-1)
    return dataset

def train_input_fn_v2(filenames):
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(5000, 200))
    dataset = dataset.map(lambda x: read_img(x, 1))
    if D.model == 'RDN':
        dataset = dataset.map(lambda x: tf.py_func(preprocess_img, [x], Tout=[tf.float32, tf.float32]))
    else:
        dataset = dataset.map(tf_preprocess_metaSR_img_new, num_parallel_calls=20)
    # dataset = dataset.apply(tf.data.experimental.unbatch())
    # dataset = dataset.batch(batch_size=D.batch_size)
    dataset = dataset.prefetch(-1)
    return dataset


def eval_input_fn(filenames):
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(lambda x: read_img(x, D.meta_sr_upsample_scale))
    dataset = dataset.map(preprocess_eval_image)
    dataset = dataset.prefetch(-1)
    return dataset

def test_input_fn(filenames):
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(read_test_img)
    dataset = dataset.prefetch(-1)
    return dataset


if __name__ == '__main__':
    tf.enable_eager_execution()
    train_filenames = glob.glob('/home/admin-seu/sss/Dataset/DIV2K_train_HR/*')
    test_filenames = glob.glob('/home/admin-seu/sss/Dataset/DIV2K_test_HR/*')
    # dataset = eval_input_fn(test_filenames)
    dataset = train_input_fn_v2(train_filenames)
    for value in dataset:
        # print(value)
        print(tf.shape(value[0]))
        print(tf.shape(value[1]))
