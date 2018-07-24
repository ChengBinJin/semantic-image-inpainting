import numpy as np
import tensorflow as tf
import cv2

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', '', 'model path (.pb)')
tf.flags.DEFINE_string('output', 'output_sample.png', 'output image path (.jpg)')
tf.flags.DEFINE_integer('image_size', '256', 'image size, default: 256')


def inference():
    graph = tf.Graph()

    with graph.as_default():
        with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model_file.read())

        z_vector = np.random.randn(1, 100).astype(np.float32)
        print(z_vector.shape)
        # print(z_vector)
        [output_image] = tf.import_graph_def(graph_def, input_map={'input_z': z_vector},
                                             return_elements=['gen_out:0'], name='output')

    with tf.Session(graph=graph) as sess:
        generated = output_image.eval()
        print('generated shape: {}'.format(generated.shape))
        img = generated[0]
        img = (img[:, :, ::-1] + 1.) / 2.
        img = (255. * img).astype(np.uint8)

        cv2.imwrite('image.png', img)
        cv2.imshow('image', img)
        cv2.waitKey(1)

        with open(FLAGS.output, 'wb') as f:
            f.write(generated[0])


def main(unused_argv):
    inference()


if __name__ == '__main__':
    tf.app.run()
