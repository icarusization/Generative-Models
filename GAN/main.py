import os
import scipy.misc
import numpy as np

from DCGAN import DCGAN
from utils import pp
from GUI import GUI
from Tkinter import *

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 200, "Epoch to train [25]")
flags.DEFINE_float("learning_rate_d", 0.0001, "Learning rate of for rmsprop [0.0002]")
flags.DEFINE_float("learning_rate_g", 0.0001, "Learning rate of for rmsprop [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 64, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "coco", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("is_GUI", True, "True for GUI, False for nothing [True]")
flags.DEFINE_integer("Lambda", 10, "Gradient penalty lambda hyperparameter")
flags.DEFINE_integer("d_iters",5, "Number of discriminator training steps per generator training step")
flags.DEFINE_integer("g_iters",1, "Number of generator training steps per generator training step")
flags.DEFINE_integer("y_dim",128,"Number of dimensions for y")
flags.DEFINE_integer("embedding_dim",1024,"Number of dimensions for embedding")
flags.DEFINE_string("anno", "captions_train2014.json", "The name of Annotation file")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)
    '''
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.6
    with tf.Session(config=config) as sess:
        dcgan1=DCGAN(sess,
                      image_size=FLAGS.image_size,
                      batch_size=FLAGS.batch_size,
                      output_size=FLAGS.output_size,
                      y_dim=None,
                      embedding_dim=FLAGS.embedding_dim,
                      c_dim=FLAGS.c_dim,
                      Lambda=FLAGS.Lambda,
                      dataset_name='wikiart',
                      is_crop=FLAGS.is_crop,
                      checkpoint_dir=FLAGS.checkpoint_dir,
                      sample_dir=FLAGS.sample_dir,
                      model_name='dcgan1')

        dcgan0 = DCGAN(sess,
                       image_size=FLAGS.image_size,
                       batch_size=FLAGS.batch_size,
                       output_size=FLAGS.output_size,
                       y_dim=FLAGS.y_dim,
                       embedding_dim=FLAGS.embedding_dim,
                       c_dim=FLAGS.c_dim,
                       Lambda=FLAGS.Lambda,
                       dataset_name='coco',
                       is_crop=FLAGS.is_crop,
                       checkpoint_dir=FLAGS.checkpoint_dir,
                       sample_dir=FLAGS.sample_dir,
                       model_name='dcgan0')

        init=tf.global_variables_initializer()
        sess.run(init)
        #all_vars=tf.trainable_variables()
        dcgan0.load(FLAGS.checkpoint_dir)
        dcgan1.load(FLAGS.checkpoint_dir)


        if FLAGS.is_GUI:
            root = Tk()
            myGUI = GUI(root, dcgan0,dcgan1)
            root.mainloop()

if __name__ == '__main__':
    tf.app.run()
