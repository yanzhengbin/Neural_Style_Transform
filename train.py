from __future__ import print_function
from __future__ import division
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
import reader
import model
import time
import losses
import utils
import os
import argparse


slim = tf.contrib.slim


def parse_args():
    parser = argparse.ArgumentParse()
    parser.add_argument('-c', '--conf', default='conf/mosaic.yml', help='the path to the conf file')
    return parser.parse_args()


def main(FLAGS):
    style_features_t = losses.get_style_features(FLAGS)

    # make sure the training path exits.
    training_path = os.path.join(FLAGS.model_path, FLAGS.naming)
    if not(os.path.exits(training_path)):
        os.makedirs(training_path)
    
    # build networks
    with tf.Graph().as_default():
        with tf.Session() as sess:
            network_fn = nets_factory.get_network_fn(
                FLAGS.loss_model,
                num_class=1,
                is_training=False)
            image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
                FLAGS.loss_model,
                is_training=False)
            processed_images = reader.image(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 'train2014/', image_preprocessing_fn, epochs=FLAGS.epoch)
            generated = model.net(processed_images, training=True)
            processed_generated = [image_preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size) for image in tf.unstack(generated, axis=0, num=FLAGS.batch_size)]
            processed_generated = tf.stack(processed_generated)
            _, endpoints_dict = network_fn(tf.concat([processed_generated, processed_images], 0), spatial_squeeze=False)









if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    FLAGS = utils.read_conf_file(args.conf)
    main(FLAGS)
