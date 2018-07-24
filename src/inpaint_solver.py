# ---------------------------------------------------------
# TensorFlow Semantic Image Inpainting Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
# import time
import tensorflow as tf

# from dcgan import DCGAN
# from mask_generator import gen_mask
from inpaint_model import ModelInpaint
# import tensorflow_utils as tf_utils


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.model = ModelInpaint(self.sess, self.flags)

        self._make_folders()
        self.iter_time = 0

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        # tf_utils.show_all_variables()

    def _make_folders(self):
        self.model_out_dir = "{}/model/{}".format(self.flags.dataset, self.flags.load_model)
        self.test_out_dir = "{}/test/{}".format(self.flags.dataset, self.flags.load_model)
        if not os.path.isdir(self.test_out_dir):
            os.makedirs(self.test_out_dir)

    def test(self):
        if self.load_model():
            print(' [*] Load SUCCESS!')
        else:
            print(' [!] Load Failed...')

        # num_iters = 10
        # total_time = 0.
        # for iter_time in range(num_iters):
        #     masks = gen_mask(self.flags)
        #
        #     # measure inference time
        #     start_time = time.time()
        #     imgs = self.model.sample_imgs()  # inference
        #     total_time += time.time() - start_time
        #     self.model.plots(imgs, iter_time, self.test_out_dir)
        #
        # print('Avg PT: {:.2f} msec.'.format(total_time / num_iters * 1000.))

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
