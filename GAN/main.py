import os
import scipy.misc
import numpy as np

from DCGAN import DCGAN
from utils import pp
from GUI import GUI
from Tkinter import *
from PIL import ImageTk, Image

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 200, "Epoch to train [25]")
flags.DEFINE_float("learning_rate_d", 0.00005, "Learning rate of for rmsprop [0.0002]")
flags.DEFINE_float("learning_rate_g", 0.00005, "Learning rate of for rmsprop [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 64, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "coco", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
#flags.DEFINE_float("clip_value", 0.01, "Value to which to clip the discriminator weights[0.01]")
#flags.DEFINE_integer("clip_per",1, "Experimental. Clip discriminator weights every this many steps. Only works reliably if clip_per=<d_iters")
flags.DEFINE_integer("d_iters",1, "Number of discriminator training steps per generator training step")
flags.DEFINE_integer("g_iters",2, "Number of generator training steps per generator training step")
flags.DEFINE_integer("y_dim",128,"Number of dimensions for y")
flags.DEFINE_integer("embedding_dim",1024,"Number of dimensions for embedding")
flags.DEFINE_string("anno", "captions_train2014.json", "The name of Annotation file")

flags.DEFINE_boolean("is_GUI", True, "True for GUI, False for nothing [True]")

FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        dcgan = DCGAN(sess,
                      image_size=FLAGS.image_size,
                      batch_size=FLAGS.batch_size,
                      output_size=FLAGS.output_size,
                      y_dim=FLAGS.y_dim,
                      embedding_dim=FLAGS.embedding_dim,
                      c_dim=FLAGS.c_dim,
                      dataset_name=FLAGS.dataset,
                      is_crop=FLAGS.is_crop,
                      checkpoint_dir=FLAGS.checkpoint_dir,
                      sample_dir=FLAGS.sample_dir)

        if FLAGS.is_train:
            dcgan.train(FLAGS)
        else:
            dcgan.load(FLAGS.checkpoint_dir)

        if FLAGS.is_GUI:
            root = Tk()
            #img = ImageTk.PhotoImage(Image.open(path).resize((64,64)))
            my_gui = GUI(root, wgan)

            root.mainloop()


if __name__ == '__main__':
    tf.app.run()
