import cv2
import numpy as np
import tensorflow as tf
from scipy.signal import convolve2d

from dcgan import DCGAN
from mask_generator import gen_mask


class ModelInpaint(object):
    def __init__(self, sess, flags):
        self.sess = sess
        self.flags = flags
        self.image_size = (flags.img_size, flags.img_size, 3)

        self.dcgan = DCGAN(sess, Flags(flags), self.image_size)
        self.nsize = 7  # neighboring for distance for weighted mask
        self.masks = gen_mask(self.flags)

        self._build_net()
        self._preprocess(use_weighted_mask=True)

        print('Hello ModelInpaint Class!')

    def _build_net(self):
        # self.wmasks_ph = tf.placeholder(tf.float32, [None, *self.image_size], name='wmasks')
        # self.images_ph = tf.placeholder(tf.float32, [None, *self.image_size], name='images')
        #
        # self.context_loss = tf.reduce_sum(tf.contrib.layers.flatten(
        #     tf.abs(tf.multiply(self.wmasks_ph, self.dcgan.g_samples) - tf.multiply(self.wmasks_ph, self.images_ph))), 1)
        # self.prior_loss = self.flags.lamb * self.dcgan.g_loss_without_mean
        # self.total_loss = self.context_loss + self.prior_loss
        #
        # self.grad_op = tf.gradients(self.total_loss, self.dcgan.z)

        print('Hello _build_net!')

    def _preprocess(self, use_weighted_mask=True, nsize=7):
        if use_weighted_mask:
            wmasks = self.create_weighted_mask(self.masks, nsize)
        else:
            wmasks = self.masks

        self.wmasks = self.create3_channel_masks(wmasks)
        self.masks = self.create3_channel_masks(self.masks)

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

    def check_masks(self):
        masks = gen_mask(self.flags)

        for idx in range(masks.shape[0]):
            print('idx: {}'.format(idx))
            mask = masks[idx]

            cv2.imshow('Mask', mask)
            cv2.waitKey(0)

    @staticmethod
    def create3_channel_masks(masks):
        masks_3c = np.zeros((*masks.shape, 3), dtype=np.float32)

        for idx in range(masks.shape[0]):
            mask = masks[idx]
            masks_3c[idx, :, :, :] = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        return masks_3c


class Flags(object):
    def __init__(self, flags):
        self.z_dim = flags.z_dim
        self.learning_rate = flags.learning_rate
        self.beta1 = flags.momentum
        # self.batch_size = 256
        # self.sample_batch = flags.sample_batch
        # self.print_freq = flags.print_freq
        # self.iters = flags.iters
        # self.dataset = flags.dataset
        # self.gpu_index = flags.gpu_index
