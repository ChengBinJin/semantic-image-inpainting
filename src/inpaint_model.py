import collections
import numpy as np
import tensorflow as tf
from scipy.signal import convolve2d
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want to solve Segmentation fault (core dumped)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import utils as utils
from dcgan import DCGAN
from mask_generator import gen_mask


class ModelInpaint(object):
    def __init__(self, sess, flags):
        self.sess = sess
        self.flags = flags
        self.image_size = (flags.img_size, flags.img_size, 3)

        self.z_vectors, self.learning_rate, self.velocity = None, None, None
        self.masks, self.wmasks = None, None

        self.dcgan = DCGAN(sess, Flags(flags), self.image_size)
        self._build_net()
        self._tensorboard()

        print('Initialized Model Inpaint SUCCESS!')

    def _build_net(self):
        self.wmasks_ph = tf.placeholder(tf.float32, [None, *self.image_size], name='wmasks')
        self.images_ph = tf.placeholder(tf.float32, [None, *self.image_size], name='images')

        self.context_loss = tf.reduce_sum(tf.contrib.layers.flatten(
            tf.abs(tf.multiply(self.wmasks_ph, self.dcgan.g_samples) - tf.multiply(self.wmasks_ph, self.images_ph))), 1)
        self.prior_loss = tf.squeeze(self.flags.lamb * self.dcgan.g_loss_without_mean)  # from (2, 1) to (2,)
        self.total_loss = self.context_loss + self.prior_loss

        self.grad = tf.gradients(self.total_loss, self.dcgan.z)

    def preprocess(self, use_weighted_mask=True, nsize=7):
        self.z_vectors = np.random.randn(self.flags.sample_batch, self.flags.z_dim)
        self.masks = gen_mask(self.flags)
        self.learning_rate = self.flags.learning_rate
        self.velocity = 0.  # for latent vector optimization

        if use_weighted_mask is True:
            wmasks = self.create_weighted_mask(self.masks, nsize)
        else:
            wmasks = self.masks

        self.wmasks = self.create3_channel_masks(wmasks)
        self.masks = self.create3_channel_masks(self.masks)

    def _tensorboard(self):
        tf.summary.scalar('loss/context_loss', tf.reduce_mean(self.context_loss))
        tf.summary.scalar('loss/prior_loss', tf.reduce_mean(self.prior_loss))
        tf.summary.scalar('loss/total_loss', tf.reduce_mean(self.total_loss))

        self.summary_op = tf.summary.merge_all()

    def __call__(self, imgs, iter_time):
        feed_dict = {self.dcgan.z: self.z_vectors,
                     self.wmasks_ph: self.wmasks,
                     self.images_ph: imgs}
        out_vars = [self.context_loss, self.prior_loss, self.total_loss, self.grad, self.dcgan.g_samples,
                    self.summary_op]

        context_loss, prior_loss, total_loss, grad, img_out, summary = self.sess.run(out_vars, feed_dict=feed_dict)

        # learning rate control
        if np.mod(iter_time, 100) == 0:
            self.learning_rate *= 0.95

        # Nesterov Acceleratd Gradient (NAG)
        v_prev = np.copy(self.velocity)
        self.velocity = self.flags.momentum * self.velocity - self.learning_rate * grad[0]
        self.z_vectors += -self.flags.momentum * v_prev + (1 + self.flags.momentum) * self.velocity
        self.z_vectors = np.clip(self.z_vectors, -1., 1.)  # as paper mentioned

        return [context_loss, prior_loss, total_loss], img_out, summary

    def print_info(self, loss, iter_time, num_try):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([('num_try', num_try), ('tar_try', self.flags.num_try),
                                                  ('cur_iter', iter_time), ('tar_iters', self.flags.iters),
                                                  ('batch_size', self.flags.sample_batch),
                                                  ('context_loss', np.mean(loss[0])),
                                                  ('prior_loss', np.mean(loss[1])),
                                                  ('total_loss', np.mean(loss[2])),
                                                  ('mask_type', self.flags.mask_type),
                                                  ('gpu_index', self.flags.gpu_index)])

            utils.print_metrics(iter_time, ord_output)

    @staticmethod
    def create_weighted_mask(masks, nsize):
        wmasks = np.zeros_like(masks)
        ker = np.ones((nsize, nsize), dtype=np.float32)
        ker = ker / np.sum(ker)

        for idx in range(masks.shape[0]):
            mask = masks[idx]
            inv_mask = 1. - mask
            temp = mask * convolve2d(inv_mask, ker, mode='same', boundary='symm')
            wmasks[idx] = mask * temp

        return wmasks

    @staticmethod
    def create3_channel_masks(masks):
        masks_3c = np.zeros((*masks.shape, 3), dtype=np.float32)

        for idx in range(masks.shape[0]):
            mask = masks[idx]
            masks_3c[idx, :, :, :] = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        return masks_3c

    def plots(self, img_list, save_file, num_try):
        n_cols = len(img_list)
        n_rows = self.flags.sample_batch

        # parameters for plot size
        scale, margin = 0.04, 0.001
        cell_size_h, cell_size_w = img_list[0][0].shape[0] * scale, img_list[0][0].shape[1] * scale
        fig = plt.figure(figsize=(cell_size_w * n_cols, cell_size_h * n_rows))  # (column, row)
        gs = gridspec.GridSpec(n_rows, n_cols)  # (row, column)
        gs.update(wspace=margin, hspace=margin)

        # save more bigger image
        for col_index in range(n_cols):
            for row_index in range(n_rows):
                ax = plt.subplot(gs[row_index * n_cols + col_index])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')

                if col_index == 0:  # original input image
                    plt.imshow((img_list[col_index][row_index] * self.masks[row_index]).reshape(
                        self.image_size[0], self.image_size[1], self.image_size[2]), cmap='Greys_r')
                else:
                    plt.imshow((img_list[col_index][row_index]).reshape(
                        self.image_size[0], self.image_size[1], self.image_size[2]), cmap='Greys_r')

        plt.savefig(save_file + '/{}_{}.png'.format(self.flags.mask_type, num_try), bbox_inches='tight')
        plt.close(fig)


class Flags(object):
    def __init__(self, flags):
        self.z_dim = flags.z_dim
        self.learning_rate = flags.learning_rate
        self.beta1 = flags.momentum
        self.sample_batch = flags.sample_batch
