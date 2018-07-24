# ---------------------------------------------------------
# Python Utility Function Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import sys
import random
import scipy.misc
import numpy as np
from PIL import Image


class ImagePool(object):
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.imgs = []

    def query(self, img):
        if self.pool_size == 0:
            return img

        if len(self.imgs) < self.pool_size:
            self.imgs.append(img)
            return img
        else:
            if random.random() > 0.5:
                # use old image
                random_id = random.randrange(0, self.pool_size)
                tmp_img = self.imgs[random_id].copy()
                self.imgs[random_id] = img.copy()
                return tmp_img
            else:
                return img


def center_crop(img, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h

    h, w = img.shape[:2]
    h_start = int(round((h - crop_h) / 2.))
    w_start = int(round((w - crop_w) / 2.))
    # resize image
    img_crop = scipy.misc.imresize(img[h_start:h_start+crop_h, w_start:w_start+crop_w], [resize_h, resize_w])
    return img_crop


def imread(path, is_gray_scale=False, img_size=None):
    if is_gray_scale:
        img = scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        img = scipy.misc.imread(path, mode='RGB').astype(np.float)

        if not (img.ndim == 3 and img.shape[2] == 3):
            img = np.dstack((img, img, img))

    if img_size is not None:
        img = scipy.misc.imresize(img, img_size)

    return img


def load_data(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, is_gray_scale=False):
    img = imread(path=image_path, is_gray_scale=is_gray_scale)

    if crop:
        cropped_img = center_crop(img, input_height, input_width, resize_height, resize_width)
    else:
        cropped_img = scipy.misc.imresize(img, [resize_height, resize_width])

    img_trans = transform(cropped_img)  # from [0, 255] to [-1., 1.]
    img_flip = random_flip(img_trans)

    if is_gray_scale and (img_flip.ndim == 2):
        img_flip = np.expand_dims(img_flip, axis=2)

    return img_flip


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname)
                         for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname)
                         for fname in os.listdir(path) if fname.endswith(extension)]

    if sort:
        filenames = sorted(filenames)

    return filenames


def imagefiles2arrs(filenames):
    img_shape = image_shape(filenames[0])
    images_arr = None

    if len(img_shape) == 3:  # color image
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
    elif len(img_shape) == 2:  # gray scale image
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1]), dtype=np.float32)

    for file_index in range(len(filenames)):
        img = Image.open(filenames[file_index])
        images_arr[file_index] = np.asarray(img).astype(np.float32)

    return images_arr


def image_shape(filename):
    img = Image.open(filename, mode="r")
    img_arr = np.asarray(img)
    img_shape = img_arr.shape
    return img_shape


def print_metrics(itr, kargs):
    print("*** Iteration {}  ====> ".format(itr))
    for name, value in kargs.items():
        print("{} : {}, ".format(name, value))
    print("")
    sys.stdout.flush()


def transform(img):
    return img / 127.5 - 1.0


def inverse_transform(img):
    return (img + 1.) / 2.


def random_flip(img):
    trans_img = img.copy()
    if random.random() > 0.5:
        trans_img = img[:, ::-1, :]

    return trans_img

