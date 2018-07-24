import os
import tensorflow as tf

from dcgan import DCGAN

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')
tf.flags.DEFINE_integer('batch_size', 256, 'batch size for one feed forwrad, default: 256')
tf.flags.DEFINE_string('dataset', 'celebA', 'dataset name for choice [celebA|svhn], default: celebA')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_integer('z_dim', 100, 'dimension of z vector, default: 100')
tf.flags.DEFINE_integer('iters', 200000, 'number of iterations, default: 200000')
tf.flags.DEFINE_integer('print_freq', 100, 'print frequency for loss, default: 100')
tf.flags.DEFINE_integer('sample_batch', 64, 'number of sampling images for check generator quality, default: 64')
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to test, (e.g. 20180704-1736), default: None')


def export_graph(model_name, image_size=(64, 64, 3)):
    graph = tf.Graph()

    with graph.as_default():
        dcgan = DCGAN(None, FLAGS, image_size)

        gen_in = tf.placeholder(tf.float32, shape=[None, FLAGS.z_dim], name='gen_in')
        gen_out = dcgan.generator(gen_in, is_reuse=True)
        gen_loss = dcgan.g_loss(gen_in)

        gen_out = tf.identity(gen_out, name='gen_out')
        restore_saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        model_out_dir = os.path.join(FLAGS.dataset, 'model', FLAGS.load_model)
        load_model(sess, restore_saver, model_out_dir)

        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), [gen_out.op.name])

        tf.train.write_graph(output_graph_def, 'pretrained', model_name, as_text=False)


def load_model(sess, saver, model_out_dir):
    print(' [*] Reading checkpoint...')

    ckpt = tf.train.get_checkpoint_state(model_out_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(model_out_dir, ckpt_name))

        meta_graph_path = ckpt.model_checkpoint_path + '.meta'
        iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

        print('===========================')
        print('   iter_time: {}'.format(iter_time))
        print('===========================')
        return True
    else:
        return False


def main(_):
    image_size = (64, 64, 3)

    print('Export DCGAN model...')
    export_graph(FLAGS.dataset + '_dcgan.pb', image_size)


if __name__ == '__main__':
    tf.app.run()
