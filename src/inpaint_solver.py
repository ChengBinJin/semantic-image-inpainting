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
# import tensorflow_utils as tf_utils


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.dataset = Dataset(self.flags, self.flags.dataset)
        self.model = ModelInpaint(self.sess, self.flags)

        self._make_folders()
        self.img_list = []

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        # tf_utils.show_all_variables()

    def _make_folders(self):
        self.model_out_dir = "{}/model/{}".format(self.flags.dataset, self.flags.load_model)
        self.test_out_dir = "{}/test/{}".format(self.flags.dataset, self.flags.load_model)
        if not os.path.isdir(self.test_out_dir):
            os.makedirs(self.test_out_dir)

        self.train_writer = tf.summary.FileWriter("{}/logs/{}".format(self.flags.dataset, self.flags.load_model),
                                                  graph_def=self.sess.graph_def)

    def test(self):
        if self.load_model():
            print(' [*] Load SUCCESS!')
        else:
            print(' [!] Load Failed...')

        imgs = self.dataset.val_next_batch(batch_size=self.flags.sample_batch)
        self.img_list.append(imgs * self.model.masks)  # save masked images

        start_time = time.time()  # measure inference time
        for iter_time in range(self.flags.iters):
            loss, img_out, summary = self.model(imgs)  # inference
            self.sample(iter_time, img_out)  # save interval results

            self.model.print_info(loss, iter_time)  # pring loss information
            self.train_writer.add_summary(summary, iter_time)  # write to tensorboard
            self.train_writer.flush()

        total_time = time.time() - start_time
        print('Total PT: {:.2f} msec.'.format(total_time * 1000.))

        self.img_list.append(imgs)  # save GT images
        self.model.plots(self.img_list, self.test_out_dir)  # save all of the images

    def sample(self, iter_time, img_out):
        if np.mod(iter_time, self.flags.sample_freq) == 0:
            self.img_list.append(img_out)

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
