from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *
from tensorflow.contrib import losses
from embedding import tools
import json

class DCGAN(object):
    def __init__(self, sess, image_size=108, is_crop=True,
                 batch_size=64, sample_size=64, output_size=64,
                 y_dim=128,embedding_dim=1024, z_dim=100, gf_dim=64, df_dim=64,
                c_dim=3, Lambda=10,dataset_name='default',
                 checkpoint_dir=None, sample_dir=None, model_name=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in last conv layer. [128]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.y_dim = y_dim  # 128
        self.embedding_dim=embedding_dim #1024
        self.z_dim = z_dim  # 100

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.c_dim = c_dim

        self.Lambda=Lambda

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.model_name=model_name
        with tf.variable_scope(self.model_name) as scope:
        	'''
        	self.d_bn1 = batch_norm(name='d_bn1')
        	self.d_bn2 = batch_norm(name='d_bn2')
        	self.d_bn3 = batch_norm(name='d_bn3')
        	self.d_bn4 = batch_norm(name='d_bn4')
        	self.d_bn5 = batch_norm(name='d_bn5')
			'''
        	self.g_bn0 = batch_norm(name='g_bn0')
        	self.g_bn1 = batch_norm(name='g_bn1')
        	self.g_bn2 = batch_norm(name='g_bn2')
        	self.g_bn3 = batch_norm(name='g_bn3')

        	self.build_model()

    def build_model(self):
        if self.y_dim:
            self.embedding = tf.placeholder(tf.float32, [None, self.embedding_dim], name='correct_embedding')
            self.embedding_fake = tf.placeholder(tf.float32, [self.batch_size, self.embedding_dim], name='wrong_embedding')

        self.images = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, self.c_dim],
                                     name='real_images')
        self.sample_images = tf.placeholder(tf.float32,
                                            [self.sample_size] + [self.output_size, self.output_size, self.c_dim],
                                            name='sample_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim],
                                name='z')

        self.z_sum = histogram_summary("z", self.z)

        if self.y_dim:
            self.G = self.generator(self.z, self.embedding)
            self.D, self.D_logits = self.discriminator(self.images, self.embedding, reuse=False)
            self.D_im, self.D_logits_im = self.discriminator(self.G, self.embedding, reuse=True)
            self.D_la,self.D_logits_la= self.discriminator(self.images, self.embedding_fake, reuse=True)
            self.sampler = self.sampler(self.z, self.embedding)
        else:
            self.G = self.generator(self.z)
            self.D, self.D_logits = self.discriminator(self.images)
            self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
            self.sampler = self.sampler(self.z)

        self.d_sum = histogram_summary("d", self.D)
        self.G_sum = image_summary("G", self.G)
        if self.y_dim:
            self.d_im_sum = histogram_summary("d_im", self.D_im)
            self.d_la_sum = histogram_summary("d_la", self.D_la)
        else:
            self.d__sum=histogram_summary("d_im", self.D_)

        '''
        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake_im = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_im, tf.zeros_like(self.D_im)))

        if self.y_dim:
            self.d_loss_fake_la = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.D_logits_la, tf.zeros_like(self.D_la)))
        else:
            self.d_loss_fake_la = 0

        self.d_loss_fake=0.5*(self.d_loss_fake_im+self.d_loss_fake_la)
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss = tf.reduce_mean(                                                                                                                                                                                                                                                                 
            sigmoid_cross_entropy_with_logits(self.D_logits_im, tf.ones_like(self.D_im)))
		'''
		
        self.d_loss_real = tf.reduce_mean(self.D_logits)
        if self.y_dim:
            self.d_loss_fake_im = tf.reduce_mean(self.D_logits_im)
            self.d_loss_fake_la = tf.reduce_mean(self.D_logits_la)
            self.d_loss_fake = 0.5 * (self.d_loss_fake_im + self.d_loss_fake_la)
            self.g_loss = -self.d_loss_fake_im
        else:
            self.d_loss_fake = tf.reduce_mean(self.D_logits_)
            self.g_loss = -self.d_loss_fake
        self.d_loss = -self.d_loss_real + self.d_loss_fake
        

        #add the gradient penalty term
        alpha=tf.random_uniform(shape=[self.batch_size,1],minval=0,maxval=1)
        differences=self.G-self.images
        interpolates=self.images+(alpha*differences)
        if self.y_dim:
            gradients=tf.gradients(self.discriminator(interpolates, self.embedding, reuse=True)[1],[interpolates])[0]
        else:    
            gradients=tf.gradients(self.discriminator(interpolates, reuse=True)[1],[interpolates])[0]
        slopes=tf.sqrt(tf.reduce_sum(tf.square(gradients),reduction_indices=[1,2,3]))
        gradient_penalty=tf.reduce_mean((slopes-1.)**2)
        self.d_loss+=self.Lambda*gradient_penalty

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum=scalar_summary("d_loss_fake", self.d_loss_fake)
        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        if self.y_dim:
            self.d_loss_fake_im_sum = scalar_summary("d_loss_fake_im", self.d_loss_fake_im)
            self.d_loss_fake_la_sum = scalar_summary("d_loss_fake_la", self.d_loss_fake_la)

        t_vars = tf.trainable_variables()
        model_vars= [var for var in t_vars if self.model_name in var.name]
        self.d_vars = [var for var in model_vars if 'd_' in var.name]
        self.g_vars = [var for var in model_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(model_vars)

    def train(self, config):
        """Train DCGAN"""
        if config.dataset == 'mnist':
            data_X, data_y = self.load_mnist()
        else:
            data = glob(os.path.join("./data", config.dataset, "*.jpg"))
        # np.random.shuffle(data)

        d_optim = tf.train.AdamOptimizer(config.learning_rate_d, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate_g, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        try:
            init_op = tf.global_variables_initializer()
        except:
            init_op = tf.initialize_all_variables()

        self.sess.run(init_op)

        if self.y_dim:
        	self.g_sum = merge_summary([self.z_sum, self.d_im_sum,self.d_la_sum,
                                    self.G_sum, self.d_loss_fake_im_sum,
                                    self.d_loss_fake_la_sum,self.g_loss_sum])
        else:
        	self.g_sum = merge_summary([self.z_sum, self.d__sum,
                                    self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        	
        self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim))

        if self.y_dim:
            anno_dir = os.path.join("./data", config.dataset, config.anno)
            embedding_model = tools.load_model()
            with open(anno_dir, 'r') as f:
                captions=json.load(f)
                annotations=captions['annotations']
                caption_dict={x['image_id']:x['caption'] for x in annotations}
                ids,captions=caption_dict.keys(),caption_dict.values()
                embeddings=tools.encode_sentences(embedding_model,X=captions, verbose=False)
                embedding_dict=dict(zip(ids,embeddings))
            print "Embedding finished."

        sample_files = data[0:self.sample_size]
        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_size,
                            is_grayscale=self.is_grayscale) for sample_file in sample_files]
        
        if self.y_dim:
            sample_ids = [int(x[27:-4]) for x in sample_files]
            sample_labels = [embedding_dict[x] for x in sample_ids]
            #sample_captions=[caption_dict[x] for x in sample_ids]

        if (self.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            if config.dataset == 'mnist':
                batch_idxs = min(len(data_X), config.train_size) // config.batch_size
            else:
                data = glob(os.path.join("./data", config.dataset, "*.jpg"))
                batch_idxs = min(len(data), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs-1):
                batch_files = data[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_size,
                                   is_grayscale=self.is_grayscale) for batch_file in batch_files]
                if self.y_dim:
                    
                    batch_ids = [int(x[27:-4]) for x in batch_files]
                    batch_labels = [embedding_dict[x] for x in batch_ids]

                    batch_files_fake=data[(idx+1) * config.batch_size:(idx + 2) * config.batch_size]
                    batch_ids_fake = [int(x[27:-4]) for x in batch_files_fake]
                    batch_labels_fake=[embedding_dict[x] for x in batch_ids_fake]


                if (self.is_grayscale):
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                    .astype(np.float32)

                # Update D network
                for iter in range(config.d_iters):
                    if self.y_dim:
                        _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                   feed_dict={self.images: batch_images, self.z: batch_z,
                                                              self.embedding: batch_labels,
                                                              self.embedding_fake:batch_labels_fake})
                    else:
                        _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                   feed_dict={self.images: batch_images, self.z: batch_z})
                    self.writer.add_summary(summary_str, counter)

                # Update G network
                for iter in range(config.g_iters):
                    if self.y_dim:
                        _, summary_str = self.sess.run([g_optim, self.g_sum],
                                        feed_dict={self.z: batch_z,self.embedding: batch_labels,
                                                   self.images: batch_images,
                                                   self.embedding_fake: batch_labels_fake})
                        self.writer.add_summary(summary_str, counter)


                    else:
                        _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={self.z: batch_z})
                        self.writer.add_summary(summary_str, counter)

                #Loss Evaluation
                if self.y_dim:
                    errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.embedding: batch_labels,
                                                       self.images: batch_images,
                                                       self.embedding_fake: batch_labels_fake})
                    errD_real = self.d_loss_real.eval({self.images: batch_images, self.embedding: batch_labels})
                    errG = self.g_loss.eval({self.z: batch_z, self.embedding: batch_labels})
                else:
                    errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                    errD_real = self.d_loss_real.eval({self.images: batch_images})
                    errG = self.g_loss.eval({self.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                    time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, 100) == 1:
                    if self.y_dim:
                        samples = self.sess.run(self.sampler,
                                                feed_dict={self.z: sample_z,
                                                            self.embedding: sample_labels})
                    else:
                        samples = self.sess.run(self.sampler,
                                                feed_dict={self.z: sample_z})
                    save_images(samples, [8, 8],
                        './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                    #print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            #h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            #h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            #h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            h1 = lrelu(conv2d(h0, self.df_dim * 2, name='d_h1_conv'))
            h2 = lrelu(conv2d(h1, self.df_dim * 4, name='d_h2_conv'))
            h3 = lrelu(conv2d(h2, self.df_dim * 8, name='d_h3_conv'))
            
            if self.y_dim:
                # When the spatial dimension is 4*4, replicate the description embedding spatially
                # perform a depth concatenation
                y_compressed = lrelu(linear(y, self.y_dim, 'd_em_to_y'))
                yb = tf.reshape(y_compressed, [self.batch_size, 1, 1, self.y_dim])
                h3= conv_cond_concat(h3, yb)

            #h4=lrelu(self.d_bn4(conv2d(h3,self.df_dim*8,k_h=1,k_w=1,d_h=1,d_w=1,name='d_h4_conv')))
            #h5=self.d_bn5(conv2d(h4,1,k_h=4,k_w=4,d_h=1,d_w=1,name='d_h5_conv'))
            h4=lrelu(conv2d(h3,self.df_dim*8,k_h=1,k_w=1,d_h=1,d_w=1,name='d_h4_conv'))
            h5=conv2d(h4,1,k_h=4,k_w=4,d_h=1,d_w=1,name='d_h5_conv')

            return tf.nn.sigmoid(h5), h5

    def generator(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            s = self.output_size
            s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)
            #32,16,8,4

            if self.y_dim:
                #concatenate y to the noise vector z
                y_compressed = lrelu(linear(y, self.y_dim, 'g_em_to_y'))
                z = concat([z, y_compressed], 1)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 8 * s16 * s16, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(self.z_, [-1, s16, s16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(h0,
                                                     [self.batch_size, s8, s8, self.gf_dim * 4], name='g_h1',
                                                     with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(h1,
                                                [self.batch_size, s4, s4, self.gf_dim * 2], name='g_h2',
                                                with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(h2,
                                                [self.batch_size, s2, s2, self.gf_dim * 1], name='g_h3',
                                                with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(h3,
                                                [self.batch_size, s, s, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)

    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s = self.output_size
            s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

            if self.y_dim:
                #concatenate y to the noise vector z
                y_compressed = lrelu(linear(y, self.y_dim, 'g_em_to_y'))
                z = concat([z, y_compressed], 1)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 8 * s16 * s16, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(self.z_, [-1, s16, s16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(h0,
                                                     [self.batch_size, s8, s8, self.gf_dim * 4], name='g_h1',
                                                     with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(h1,
                                                [self.batch_size, s4, s4, self.gf_dim * 2], name='g_h2',
                                                with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(h2,
                                                [self.batch_size, s2, s2, self.gf_dim * 1], name='g_h3',
                                                with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(h3,
                                                [self.batch_size, s, s, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)
                


    def save(self, checkpoint_dir, step):
        model_name = self.model_name
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def display(self, embeddings=None, z=None):
        result = None
        sample_z = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim))
        if z!=None:
            for eachz in sample_z:
                eachz[0:len(z)] = z
        if embeddings!=None:
            embedding4feed=[]
            for i in xrange(self.sample_size):
                embedding4feed.append(embeddings[0])
            result = self.sess.run(self.sampler,feed_dict={self.z: sample_z,self.embedding: embedding4feed})
        else:
            result = self.sess.run(self.sampler,feed_dict={self.z: sample_z})
        return result


    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False
