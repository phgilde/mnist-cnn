import sys

import numpy as np
import pygame as pg
from scipy.ndimage import center_of_mass
from scipy.ndimage.interpolation import shift
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


successes, failures = pg.init()
print("{0} successes and {1} failures".format(successes, failures))

px_size = 20

WIDTH, HEIGHT = 28 * px_size, 29 * px_size

screen = pg.display.set_mode((WIDTH, HEIGHT))
clock = pg.time.Clock()

image = np.zeros((28, 28))


height = 28
width = 28
channels = 1
n_inputs = height * width

conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 2
conv2_pad = "SAME"

pool3_fmaps = conv2_fmaps

n_fc1 = 64
n_outputs = 10

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")

conv1 = tf.layers.conv2d(
    X_reshaped,
    filters=conv1_fmaps,
    kernel_size=conv1_ksize,
    strides=conv1_stride,
    padding=conv1_pad,
    activation=tf.nn.relu,
    name="conv1",
)
conv2 = tf.layers.conv2d(
    conv1,
    filters=conv2_fmaps,
    kernel_size=conv2_ksize,
    strides=conv2_stride,
    padding=conv2_pad,
    activation=tf.nn.relu,
    name="conv2",
)

with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(
        conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID"
    )
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 7 * 7])

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=y
    )
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

with tf.Session() as sess:
    last_result = 1
    saver.restore(sess, "mnist_cnn_model/my_mnist_model")
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

        for x in range(28):
            for y in range(28):
                surface = pg.Surface((px_size, px_size))
                surface.fill([int(image[x, y] * 255) for i in range(3)])
                # surface.fill((255, 255, 255))
                # print([int(image[x, y]*255) for i in range(3)])
                rect = pg.Rect(x * px_size, y * px_size, px_size, px_size)
                screen.blit(surface, rect)
        x_mouse, y_mouse = pg.mouse.get_pos()

        if pg.mouse.get_pressed()[0]:
            if y_mouse < 28 * px_size:
                x_im, y_im = (x_mouse // px_size), (y_mouse // px_size)
                image[x_im, y_im] += 0.2
                if image[x_im, y_im] > 1:
                    image[x_im, y_im] = 1
            else:
                image = np.zeros((28, 28))

        pg.display.update()
        image_rot = np.fliplr(np.rot90(image, 3))
        c = center_of_mass(image_rot)
        offset_x, offset_y = 14 - c[0], 14 - c[1]
        image_shift = shift(image_rot, (offset_x, offset_y), cval=0, mode="constant")
        y_logits = logits.eval(feed_dict={X: image_shift.reshape(1, -1)})
        result = np.argmax(y_logits)
        if result != last_result:
            print(result)

            last_result = result
