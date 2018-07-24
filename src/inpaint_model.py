from dcgan import DCGAN


class ModelInpaint(object):
    def __init__(self, sess, flags):
        self.sess = sess
        self.flags = flags
        self.image_size = (flags.img_size, flags.img_size, 3)

        self.dcgan = DCGAN(sess, Flags(flags), self.image_size)
        print('Hello ModelInpaint Class!')


class Flags(object):
    def __init__(self, flags):
        self.z_dim = flags.z_dim
        self.learning_rate = flags.learning_rate
        self.beta1 = 0.5
        self.batch_size = 256
        self.sample_batch = flags.sample_batch
        self.print_freq = flags.print_freq
        self.iters = flags.iters
        self.dataset = flags.dataset
        self.gpu_index = flags.gpu_index
