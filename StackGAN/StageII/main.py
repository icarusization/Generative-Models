import os
import scipy.misc
import numpy as np

from StackGAN import StackGAN
from utils import pp
#from GUI import GUI
from Tkinter import *

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 600, "Epoch to train [25]")
flags.DEFINE_float("learning_rate_d", 0.0001, "Learning rate of for rmsprop [0.0002]")
flags.DEFINE_float("learning_rate_g", 0.0001, "Learning rate of for rmsprop [0.0002]")
flags.DEFINE_float("lr_decay_step", 100, "number of epochs when learning rate is decayed by 1/2")
flags.DEFINE_float("is_CA",True,"True for caption augmentation")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 256, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 256, "The size of the output images to produce [64]")
flags.DEFINE_integer("lr_size", 64, "The size of the low-resolution images")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "coco_256", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
#flags.DEFINE_string("model_lr_name", "stackgan_stage1", "Name for the stage1 model")
flags.DEFINE_string("model_lr_dir", "model", "Directory name for the stage1 model")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_Wasserstein", False, "True for WGAN, False for DCGAN [False]")
#flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
#flags.DEFINE_boolean("is_GUI", False, "True for GUI, False for nothing [True]")
flags.DEFINE_integer("Lambda", 10, "Gradient penalty lambda hyperparameter")
flags.DEFINE_float("Alpha",1.0,"Weight for fake-label loss")
flags.DEFINE_integer("d_pre_train_step",0,"Number of steps for pre-training the discriminator")
flags.DEFINE_integer("d_iters",1, "Number of discriminator training steps per generator training step")
flags.DEFINE_integer("g_iters",2, "Number of generator training steps per generator training step")
flags.DEFINE_integer("y_dim",128,"Number of dimensions for y")
flags.DEFINE_integer("embedding_dim",1024,"Number of dimensions for embedding")
flags.DEFINE_string("anno", "captions_train2014.json", "The name of Annotation file")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8
    with tf.Session(config=config) as sess:
        stackgan = StackGAN(sess,
                      image_size=FLAGS.image_size,
                      batch_size=FLAGS.batch_size,
                      output_size=FLAGS.output_size,
                      lr_size=FLAGS.lr_size,
                      y_dim=FLAGS.y_dim,
                      embedding_dim=FLAGS.embedding_dim,
                      c_dim=FLAGS.c_dim,
                      Lambda=FLAGS.Lambda,
                      Alpha=FLAGS.Alpha,
                      dataset_name=FLAGS.dataset,
                      is_crop=FLAGS.is_crop,
                      is_CA=FLAGS.is_CA,
                      is_Wasserstein=FLAGS.is_Wasserstein,
                      checkpoint_dir=FLAGS.checkpoint_dir,
                      sample_dir=FLAGS.sample_dir,
                      model_lr_dir=FLAGS.model_lr_dir,
                      model_name='stackgan')
        #stackgan.load_stage1()

        
        if FLAGS.is_train:
            stackgan.train(FLAGS)
        else:
            stackgan.load(FLAGS.checkpoint_dir)
        '''
        if FLAGS.is_GUI:
            root = Tk()
            myGUI = GUI(root, dcgan)
            root.mainloop()
		    '''

if __name__ == '__main__':
    tf.app.run()
