# ---------------------------------------------------------
# TensorFlow Semantic Image Inpainting Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import time
import numpy as np
import tensorflow as tf

from dataset import Dataset
from inpaint_model import ModelInpaint
import poissonblending as poisson
import utils as utils


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.dataset = Dataset(self.flags, self.flags.dataset)
        self.model = ModelInpaint(self.sess, self.flags)

        self._make_folders()
        self.iter_time = 0

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def _make_folders(self):
        self.model_out_dir = "{}/model/{}".format(self.flags.dataset, self.flags.load_model)
        self.test_out_dir = "{}/inpaint/{}/is_blend_{}".format(self.flags.dataset, self.flags.load_model,
                                                               str(self.flags.is_blend))
        if not os.path.isdir(self.test_out_dir):
            os.makedirs(self.test_out_dir)

        self.train_writer = tf.summary.FileWriter("{}/inpaint/{}/log".format(self.flags.dataset, self.flags.load_model),
                                                  graph_def=self.sess.graph_def)

    def test(self):
        if self.load_model():
            print(' [*] Load SUCCESS!')
        else:
            print(' [!] Load Failed...')

        for num_try in range(self.flags.num_try):
            self.model.learning_rate = self.flags.learning_rate  # initialize learning rate
            imgs = self.dataset.val_next_batch(batch_size=self.flags.sample_batch)
            img_list = []
            img_list.append((imgs + 1.) / 2.)  # save masked images

            start_time = time.time()  # measure inference time
            for iter_time in range(self.flags.iters):
                loss, img_outs, summary = self.model(imgs, iter_time)  # inference
                blend_results = self.postprocess(imgs, img_outs, self.flags.is_blend)  # blending
                self.sample(img_list, iter_time, blend_results)  # save interval results

                self.model.print_info(loss, iter_time, num_try)  # pring loss information
                self.train_writer.add_summary(summary, iter_time)  # write to tensorboard
                self.train_writer.flush()

            total_time = time.time() - start_time
            print('Total PT: {:.3f} sec.'.format(total_time))

            img_list.append((imgs + 1.) / 2.)  # save GT images
            self.model.plots(img_list, self.test_out_dir, num_try)  # save all of the images

    def sample(self, img_list, iter_time, img_out):
        if np.mod(iter_time, self.flags.sample_freq) == 0:
            img_list.append(img_out)

    def postprocess(self, ori_imgs, gen_imgs, is_blend=True):
        outputs = np.zeros_like(ori_imgs)
        tar_imgs = np.asarray([utils.inverse_transform(img) for img in ori_imgs])
        sour_imgs = np.asarray([utils.inverse_transform(img) for img in gen_imgs])

        if is_blend is True:
            for idx in range(tar_imgs.shape[0]):
                outputs[idx] = poisson.blend(tar_imgs[idx], sour_imgs[idx],
                                             ((1. - self.model.masks[idx]) * 255.).astype(np.uint8))
        else:
            outputs = np.multiply(tar_imgs, self.model.masks) + np.multiply(sour_imgs, 1. - self.model.masks)

        return outputs

    def load_model(self):
        print(' [*] Reading checkpoint...')

        ckpt = tf.train.get_checkpoint_state(self.model_out_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_out_dir, ckpt_name))

            meta_graph_path = ckpt.model_checkpoint_path + '.meta'
            self.iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

            print('===========================')
            print('   iter_time: {}'.format(self.iter_time))
            print('===========================')
            return True
        else:
            return False
