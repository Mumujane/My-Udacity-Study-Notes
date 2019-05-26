import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/MNIST/", one_hot=True)
model_path = "saver/Conv_Gan.ckpt"

def placeholder(noise_dim, image_height, image_width, image_depth):
    """
    @Author: Ricardo Zeng
    ---------------------
    :param noise_dim: 噪声图片的size
    :param image_height: 真实图像的height
    :param image_width:  真实图像的width
    :param image_depth:  真实图像的颜色通道数
    """
    Inputs_real = tf.placeholder(tf.float32, shape = [None, image_height, image_width, image_depth], name = 'inputs_real')
    Inputs_noise = tf.placeholder(tf.float32, shape = [None, noise_dim], name = 'inputs_fake')

    return Inputs_real, Inputs_noise

def generator(noise_img, output_dim, is_train = True, alpha = 0.01):
    """
    @Author: Ricardo Zeng
    ---------------------
    """
    with tf.variable_scope("generator", reuse = (not is_train)):
        layer1 = tf.layers.dense(noise_img, 4*4*512)
        layer1 = tf.reshape(layer1, shape = [-1, 4, 4, 512])
        layer1 = tf.layers.batch_normalization(layer1, training = is_train)
        layer1 = tf.maximum(alpha * layer1, layer1)
        layer1 = tf.nn.dropout(layer1, keep_prob = 0.8)

        layer2 = tf.layers.conv2d_transpose(layer1, filters = 256, kernel_size = 4, strides = 1, padding = 'valid')
        layer2 = tf.layers.batch_normalization(layer2, training = is_train)
        layer2 = tf.maximum(alpha * layer2, layer2)
        layer2 = tf.nn.dropout(layer2, keep_prob = 0.8)

        layer3 = tf.layers.conv2d_transpose(layer2, 128, 3, strides = 2, padding = 'same')
        layer3 = tf.layers.batch_normalization(layer3, training = is_train)
        layer3 = tf.maximum(alpha * layer3, layer3)
        layer3 = tf.nn.dropout(layer3, keep_prob = 0.8)

        logits = tf.layers.conv2d_transpose(layer3, output_dim, 3, strides = 2, padding = 'same')

        outputs = tf.tanh(logits)

        return outputs

def discriminator(inputs_img, reuse = False, alpha = 0.01):
    """
    @Author: Ricardo Zeng
    ---------------------
    """
    with tf.variable_scope("discriminator", reuse = reuse):
        layer1 = tf.layers.conv2d(inputs_img, 128, 3, strides = 2, padding = 'same')
        layer1 = tf.nn.leaky_relu(layer1, alpha = alpha)
        layer1 = tf.nn.dropout(layer1, keep_prob = 0.8)

        layer2 = tf.layers.conv2d(layer1, 256, 3, strides = 2, padding = 'same')
        layer2 = tf.layers.batch_normalization(layer2, training = True)
        layer2 = tf.nn.leaky_relu(layer2, alpha = alpha)
        layer2 = tf.nn.dropout(layer2, keep_prob = 0.8)

        layer3 = tf.layers.conv2d(layer2, 512, 3, strides = 2, padding = 'same')
        layer3 = tf.layers.batch_normalization(layer3, training = True)
        layer3 = tf.nn.leaky_relu(layer3, alpha = alpha)
        layer3 = tf.nn.dropout(layer3, keep_prob = 0.8)

        logits = tf.contrib.layers.fully_connected(layer3, 1, activation_fn = None)
        output = tf.sigmoid(logits)

        return logits, output

def loss(inputs_real, inputs_noise, image_depth, smooth = 0.1):
    """
    @Author: Ricardo Zeng
    ---------------------
    """
    g_outputs = generator(inputs_noise, image_depth, is_train = True)
    d_logits_real, d_outputs_real = discriminator(inputs_real)
    d_logits_fake, d_outputs_fake = discriminator(g_outputs, reuse = True)
    
    g_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_fake , labels = tf.ones_like(d_outputs_fake) * (1 - smooth)))

    d_loss_real = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_real, labels = tf.ones_like(d_outputs_real) * (1 - smooth)))
    d_loss_fake = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_fake, labels = tf.zeros_like(d_outputs_fake)))

    d_loss = d_loss_real + d_loss_fake

    return g_loss, d_loss

def optimizer(g_loss, d_loss, betal = 0.4, learning_rate = 0.001):
    """
    @Author: Ricardo Zeng
    ---------------------
    """
    train_vars = tf.trainable_variables()
    g_vars = [var for var in train_vars if var.name.startswith("generator")]
    d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        g_opt = tf.train.AdamOptimizer(learning_rate, beta1 = betal).minimize(g_loss, var_list = g_vars)
        d_opt = tf.train.AdamOptimizer(learning_rate, beta1 = betal).minimize(d_loss, var_list = d_vars)

    return g_opt, d_opt

def plot_images(samples):
    fig, axes = plt.subplot(nrows = 1, ncols = 25, sharey = True, figsize = (50,2))
    for img, ax in zip(samples, axes):
        ax.imshow(img.reshape((28, 28)), cmap = 'Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad = 0)

def show_generator_output(sess, n_images, input_noise, output_dim):
    """
    @Author: Ricardo Zeng
    ---------------------
    @param sess: Tensorflow Session
    @param n_images: 展示的图片数量
    @param input_noise: 噪声图片
    @param output_dim: 图片的通道数
    """
    cmap = 'Greys_r'
    noise_shape = input_noise.get_shape().as_list()[-1]
    examples_noise = np.random.uniform(-1, 1, size = [n_images, noise_shape])
    samples = sess.run(generator(input_noise, output_dim, False), feed_dict = {input_noise: examples_noise})

    result = np.squeeze(samples, -1)
    return result

batch_size = 64
noise_size = 100
epochs = 5
n_samples = 25
learning_rate = 0.001
betal = 0.4

def train(noise_size, data_shape, batch_size, n_samples):
    """
    @Author: Ricardo Zeng
    ---------------------
    @param noise_size: 噪声图片维度
    @param data_shape: 真实图像的维度
    @param batch_size: ..
    @param n_samples: 显示示例图片数量
    """
    losses = []
    steps = 0
    inputs_real, inputs_noise = placeholder(noise_size, data_shape[1], data_shape[2], data_shape[3])
    g_loss, d_loss = loss(inputs_real, inputs_noise, data_shape[-1])
    g_train_opt, d_train_opt = optimizer(g_loss, d_loss, betal, learning_rate)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            for batch_i in range(mnist.train.num_examples//batch_size):
                steps += 1
                batch = mnist.train.next_batch(batch_size)
                batch_images = batch[0].reshape((batch_size, data_shape[1], data_shape[2], data_shape[3]))
                batch_images = batch_images * 2 - 1
                batch_noise = np.random.uniform(-1, 1, size = (batch_size, noise_size))

                _1 = sess.run(g_train_opt, feed_dict = {inputs_real: batch_images, inputs_noise: batch_noise})
                _2 = sess.run(d_train_opt, feed_dict = {inputs_real: batch_images, inputs_noise: batch_noise})

                if steps % 101 == 0:
                    train_loss_d = d_loss.eval({inputs_real: batch_images, inputs_noise: batch_noise})
                    train_loss_g = g_loss.eval({inputs_real: batch_images, inputs_noise: batch_noise})
                    losses.append((train_loss_d, train_loss_g))
                    samples = show_generator_output(sess, n_samples, inputs_noise, data_shape[-1])
                    plot_images(samples)
                    print("Epoch {}/{}....".format(e+1, epochs),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}...".format(train_loss_g))

if __name__ == '__main__':
    with tf.Graph().as_default():
        train(noise_size, [None, 28,28,1], batch_size, n_samples)                
