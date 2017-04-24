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

class StackGAN(object):
	def __init__(self, sess, image_size=108, is_crop=True,is_CA=False,
				 batch_size=64, output_size=64,lr_size=64,
				 y_dim=128,embedding_dim=1024, z_dim=100, gf_dim=64, df_dim=64,
				c_dim=3, Lambda=10,Alpha=10,dataset_name='default',
				 checkpoint_dir=None, sample_dir=None, 
				 model_lr_dir=None,model_name=None):
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
		self.is_CA=is_CA
		self.is_grayscale = (c_dim == 1)
		self.batch_size = batch_size
		self.image_size = image_size
		self.sample_size = batch_size
		#self.sample_size = sample_size
		self.output_size = output_size
		self.lr_size=lr_size

		self.y_dim = y_dim  # 128
		self.embedding_dim=embedding_dim #1024
		self.z_dim = z_dim  # 100

		self.gf_dim = gf_dim
		self.df_dim = df_dim

		self.c_dim = c_dim

		self.Lambda=Lambda
		self.Alpha=Alpha

		self.dataset_name = dataset_name
		self.checkpoint_dir = checkpoint_dir
		self.model_name=model_name
		self.model_lr_dir=model_lr_dir

		#stage1 generator batch norm
		self.g_lr_bn0 = batch_norm(name='g_bn0')
		self.g_lr_bn1 = batch_norm(name='g_bn1')
		self.g_lr_bn2 = batch_norm(name='g_bn2')
		self.g_lr_bn3 = batch_norm(name='g_bn3')
		self.g_lr_bn4 = batch_norm(name='g_bn4')
		self.g_lr_bn5 = batch_norm(name='g_bn5')
		self.g_lr_bn6 = batch_norm(name='g_bn6')
		self.g_lr_bn7 = batch_norm(name='g_bn7')
		self.g_lr_bn8 = batch_norm(name='g_bn8')
		self.g_lr_bn9 = batch_norm(name='g_bn9')
		
		#stage2 generator batch norm
		self.g_hr_bn0 = batch_norm(name='g_bn0')
		self.g_hr_bn1 = batch_norm(name='g_bn1')
		self.g_hr_bn2 = batch_norm(name='g_bn2')
		self.g_hr_bn3 = batch_norm(name='g_bn3')
		self.g_hr_bn4 = batch_norm(name='g_bn4')
		self.g_hr_bn5 = batch_norm(name='g_bn5')
		self.g_hr_bn6 = batch_norm(name='g_bn6')

		#with tf.variable_scope(self.model_name) as scope:
		with tf.variable_scope(self.model_name) as scope:
			self.build_model()
			#self.load_stage1()


	def load_stage1(self):
		print(" [*] Reading stage1 model...")

		model_dir = self.model_lr_dir

		ckpt = tf.train.get_checkpoint_state(model_dir)
		
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver_lr.restore(self.sess, os.path.join(model_dir, ckpt_name))
			print(" [*] Success to read {}".format(ckpt_name))
			return True
		else:
			print(" [*] Failed to find the stage1 model")
			return False


	def generator_lr(self, z, y=None):
		with tf.variable_scope("generator") as scope:
			s = self.lr_size
			s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)
			#32,16,8,4

			if self.y_dim:
				if self.is_CA:
					#condition augmentation
					mean=lrelu(linear(y,self.y_dim,"g_em_to_y_mean"))
					log_std=tf.nn.relu(linear(y,self.y_dim,"g_em_to_y_variance"))
					epsilon = tf.truncated_normal(tf.shape(mean))
					std = tf.exp(log_std)
					y_augmented=mean+std*epsilon
					#concatenate y to the noise vector z
					z = concat([z, y_augmented], 1)
				else:
					y_compressed=lrelu(linear(y,self.y_dim,"g_em_to_y_mean"))
					z = concat([z, y_compressed], 1)

			# project `z` and reshape
			self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 8 * s16 * s16, 'g_h0_lin', with_w=True)

			z_=self.g_lr_bn0(self.z_)

			self.h0 = tf.reshape(z_, [self.batch_size, s16, s16, self.gf_dim * 8])

			#residual block 0 
			with tf.variable_scope("g_r0"):
				self.r0_h0=conv2d(self.h0,self.gf_dim*2,k_h=1,k_w=1,d_h=1,d_w=1,name="h0")
				r0_h0=tf.nn.relu(self.g_lr_bn1(self.r0_h0))
				self.r0_h1=conv2d(r0_h0,self.gf_dim*2,k_h=3,k_w=3,d_h=1,d_w=1,name="h1")
				r0_h1=tf.nn.relu(self.g_lr_bn2(self.r0_h1))
				self.r0_h2=conv2d(r0_h1,self.gf_dim*8,k_h=3,k_w=3,d_h=1,d_w=1,name="h2")
				r0_h2=self.g_lr_bn3(self.r0_h2)
				self.r0_h3=tf.nn.relu(tf.add(r0_h2,self.h0))
			
			#upsampling+convolution
			self.h1, self.h1_w, self.h1_b = deconv2d(self.r0_h3,
													 [self.batch_size, s8, s8, self.gf_dim * 4], 
													 k_h=3,k_w=3,name='g_h1',with_w=True)
			h1 = self.g_lr_bn4(self.h1)

			#residual block 1
			with tf.variable_scope("g_r1"):
				self.r1_h0=conv2d(h1,self.gf_dim*1,k_h=1,k_w=1,d_h=1,d_w=1,name="h0")
				r1_h0=tf.nn.relu(self.g_lr_bn5(self.r1_h0))
				self.r1_h1=conv2d(r1_h0,self.gf_dim*1,k_h=3,k_w=3,d_h=1,d_w=1,name="h1")
				r1_h1=tf.nn.relu(self.g_lr_bn6(self.r1_h1))
				self.r1_h2=conv2d(r1_h1,self.gf_dim*4,k_h=3,k_w=3,d_h=1,d_w=1,name="h2")
				r1_h2=self.g_lr_bn7(self.r1_h2)
				self.r1_h3=tf.nn.relu(tf.add(r1_h2,h1))


			h2, self.h2_w, self.h2_b = deconv2d(self.r1_h3,
												[self.batch_size, s4, s4, self.gf_dim * 2], 
												k_h=3,k_w=3,name='g_h2',with_w=True)
			h2 = tf.nn.relu(self.g_lr_bn8(h2))

			h3, self.h3_w, self.h3_b = deconv2d(h2,
												[self.batch_size, s2, s2, self.gf_dim * 1], 
												k_h=3,k_w=3,name='g_h3',with_w=True)
			h3 = tf.nn.relu(self.g_lr_bn9(h3))

			h4, self.h4_w, self.h4_b = deconv2d(h3,
												[self.batch_size, s, s, self.c_dim], 
												k_h=3,k_w=3,name='g_h4',with_w=True)

			if self.is_CA:
				return tf.nn.tanh(h4),[mean,log_std]
			else:
				return tf.nn.tanh(h4)

	def build_model(self):
		self.embedding = tf.placeholder(tf.float32, [None, self.embedding_dim], name='correct_embedding')
		self.embedding_fake = tf.placeholder(tf.float32, [self.batch_size, self.embedding_dim], name='wrong_embedding')

		self.images = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, self.c_dim],
									 name='real_images')
		self.z = tf.placeholder(tf.float32, [None, self.z_dim],
								name='z')
		self.learning_rate_g=tf.placeholder(tf.float32,name='learning_rate_g')
		self.learning_rate_d=tf.placeholder(tf.float32,name='learning_rate_d')

		#self.z_sum = histogram_summary("z", self.z)
		self.images_lr=self.generator_lr(self.z, self.embedding)
		if self.is_CA:
			self.G,self.G_ = self.fcgenerator_hr(self.images_lr, self.embedding)
		else:
			self.G= self.fcgenerator_hr(self.images_lr, self.embedding)
		self.D, self.D_logits = self.discriminator_hr(self.images, self.embedding, reuse=False)
		self.D_im, self.D_logits_im = self.discriminator_hr(self.G, self.embedding, reuse=True)
		self.D_la,self.D_logits_la= self.discriminator_hr(self.images, self.embedding_fake, reuse=True)
		self.sampler = self.sampler_hr(self.images_lr, self.embedding)
		
		self.d_loss_real = tf.reduce_mean(self.D_logits)
		self.d_loss_fake = tf.reduce_mean(self.D_logits_im)
		self.d_loss_la = tf.reduce_mean(self.D_logits_la)
		if self.Alpha!=0:
			self.d_loss_distance = -self.d_loss_real+0.5*(self.d_loss_fake+self.Alpha*self.d_loss_la)
			self.d_loss=self.d_loss_distance
		else:
			self.d_loss_distance = -self.d_loss_real+self.d_loss_fake
			self.d_loss=self.d_loss_distance
		if self.is_CA:
			self.g_loss_kl=KL_loss(self.G_[0],self.G_[1])
			self.g_loss_im = -self.d_loss_fake
			self.g_loss=self.g_loss_kl+self.g_loss_im
		else:
			self.g_loss=-self.d_loss_fake
		

		#add the gradient penalty term
		alpha=tf.random_uniform(shape=[self.batch_size,1,1,1],minval=0,maxval=1)
		differences=self.G-self.images
		interpolates=self.images+(alpha*differences)
		gradients=tf.gradients(self.discriminator_hr(interpolates, self.embedding, reuse=True)[1],[interpolates])[0]
		slopes=tf.sqrt(tf.reduce_sum(tf.square(gradients),reduction_indices=[1,2,3]))
		self.gradient_penalty=tf.reduce_mean((slopes-1.)**2)
		self.d_loss+=self.Lambda*self.gradient_penalty

		self.d_loss_distance_sum = scalar_summary("d_loss_distance", self.d_loss_distance)
		self.gradient_penalty_sum=scalar_summary("gradient_penalty",self.gradient_penalty)
		self.g_loss_sum = scalar_summary("g_loss", self.g_loss)

		self.slopes=tf.reduce_mean(slopes)# avg slope value
		self.slopes_sum=scalar_summary("slopes",self.slopes)

		self.d_loss_la_sum = scalar_summary("d_loss_la", self.d_loss_la)
		if self.is_CA:
			self.g_loss_kl_sum = scalar_summary("g_loss_kl", self.g_loss_kl)
			self.g_loss_im_sum=scalar_summary("g_loss_im", self.g_loss_im)

		t_vars = tf.trainable_variables()
		model_vars= [var for var in t_vars if self.model_name in var.name]

		model_hr_vars=[var for var in model_vars if 'hr' in var.name]
		model_lr_vars=[var for var in model_vars if 'hr' not in var.name]
		self.d_vars = [var for var in model_hr_vars if 'd_hr' in var.name]
		self.g_vars = [var for var in model_hr_vars if 'g_hr' in var.name]

		self.saver = tf.train.Saver(model_vars)
		self.saver_lr = tf.train.Saver(model_lr_vars)

	def generator_hr(self, image_lr, y=None):
		with tf.variable_scope("generator_hr") as scope:
			#encode images
			with tf.variable_scope("down_sampling") as scope_0:
				#down_h0 64*64
				down_h0=conv2d(image_lr,self.gf_dim,k_h=3,k_w=3,d_h=1,d_w=1,name="g_hr_down_h0")
				down_h0=tf.nn.relu(down_h0)
				#down_h1 32*32
				down_h1=self.g_hr_bn0(conv2d(down_h0,self.gf_dim*2,k_h=4,k_w=4,name="g_hr_down_h1"))
				down_h1=tf.nn.relu(down_h1)
				#down_h2 16*16
				down_h2=self.g_hr_bn1(conv2d(down_h1,self.gf_dim*4,k_h=4,k_w=4,name="g_hr_down_h2"))
				down_h2=tf.nn.relu(down_h2)

			#joint_img_text
			with tf.variable_scope("joint_img_text") as scope_1:
				if self.is_CA:
					#condition augmentation
					mean=lrelu(linear(y,self.y_dim,"g_hr_em_to_y_mean"))
					log_std=tf.nn.relu(linear(y,self.y_dim,"g_hr_em_to_y_variance"))
					epsilon = tf.truncated_normal(tf.shape(mean))
					std = tf.exp(log_std)
					y_augmented=mean+std*epsilon
					yb = tf.reshape(y_augmented, [self.batch_size, 1, 1, self.y_dim])
					joint_img_text= conv_cond_concat(down_h2, yb)
					# -->16 * 16 * (128+512)
				else:
					y_compressed=lrelu(linear(y,self.y_dim,"g_hr_em_to_y_mean"))
					yb = tf.reshape(y_compressed, [self.batch_size, 1, 1, self.y_dim])
					joint_img_text= conv_cond_concat(down_h2, yb)
					# -->16 * 16 * (128+512)
				joint_img_text=self.g_hr_bn2(conv2d(joint_img_text,self.gf_dim*4,
									k_h=3,k_w=3,d_h=1,d_w=1,name="g_hr_joint_conv"))
				# -->16 * 16 * 512
				joint_img_text=tf.nn.relu(joint_img_text)

			#residual block*4
			with tf.variable_scope("residuals") as scope_2:
				r0=self.residual_block(joint_img_text,name="g_hr_r0")
				r1=self.residual_block(r0,name="g_hr_r1")
				r2=self.residual_block(r1,name="g_hr_r2")
				r3=self.residual_block(r2,name="g_hr_r3")
			
			#upsampling
			s = self.output_size#256
			s2, s4, s8 = int(s / 2), int(s / 4), int(s/8) #128,64,32
			with tf.variable_scope("upsampling") as scope_3:	
				up_h0=self.g_hr_bn3(deconv2d(r3,[self.batch_size, s8, s8, self.gf_dim * 2], 
										k_h=3,k_w=3,name='g_hr_up_h0'))
				up_h0=tf.nn.relu(up_h0)
				up_h1=self.g_hr_bn4(deconv2d(up_h0,[self.batch_size, s4, s4, self.gf_dim], 
										k_h=3,k_w=3,name='g_hr_up_h1'))
				up_h1=tf.nn.relu(up_h1)
				up_h2=self.g_hr_bn5(deconv2d(up_h1,[self.batch_size, s2, s2, self.gf_dim // 2], 
										k_h=3,k_w=3,name='g_hr_up_h2'))
				up_h2=tf.nn.relu(up_h2)
				up_h3=self.g_hr_bn6(deconv2d(up_h2,[self.batch_size, s, s, self.gf_dim // 4], 
										k_h=3,k_w=3,name='g_hr_up_h3'))
				up_h3=tf.nn.relu(up_h3)
				output=conv2d(up_h3,3,k_h=3,k_w=3,d_h=1, d_w=1,name="g_hr_output")

			if self.is_CA:
				return tf.nn.tanh(output),[mean,log_std]
			else:
				return tf.nn.tanh(output)

	def fcgenerator_hr(self, image_lr, y=None):
		with tf.variable_scope("generator_hr") as scope:
			#encode images
			with tf.variable_scope("down_sampling") as scope_0:
				image_lr_flatten=tf.reshape(image_lr,[self.batch_size,-1])
				#down_h0 64*64
				down_h0=linear(image_lr_flatten,self.gf_dim*8,scope="g_hr_down_h0")
				down_h0=tf.nn.relu(down_h0)
				#down_h1 32*32
				down_h1=linear(down_h0,self.gf_dim*8,scope="g_hr_down_h1")
				down_h1=tf.nn.relu(down_h1)
				#down_h2 16*16
				down_h2=linear(down_h1,self.gf_dim*8,scope="g_hr_down_h2")
				down_h2=tf.nn.relu(down_h2)

			#joint_img_text
			with tf.variable_scope("joint_img_text") as scope_1:
				y_compressed=tf.nn.relu(linear(y,self.y_dim,"g_hr_em_to_y_mean"))
				#yb = tf.reshape(y_compressed, [self.batch_size, 1, 1, self.y_dim])
				#joint_img_text= conv_cond_concat(down_h2, yb)
				# -->1batch_size * (128+2048)
				joint_img_text=concat([y_compressed,down_h2],1)
				# -->16 * 16 * 512
				#joint_img_text=tf.nn.relu(joint_img_text)
				r3=joint_img_text

			'''
			#residual block*4
			with tf.variable_scope("residuals") as scope_2:
				r0=self.residual_block(joint_img_text,name="g_hr_r0")
				r1=self.residual_block(r0,name="g_hr_r1")
				r2=self.residual_block(r1,name="g_hr_r2")
				r3=self.residual_block(r2,name="g_hr_r3")
			'''

			#upsampling
			with tf.variable_scope("upsampling") as scope_3:	
				up_h0=linear(r3,self.gf_dim*8,scope='g_hr_up_h0')
				up_h0=tf.nn.relu(up_h0)
				up_h1=linear(up_h0,self.gf_dim*8,scope='g_hr_up_h1')
				up_h1=tf.nn.relu(up_h1)
				up_h2=linear(up_h1,self.gf_dim*8,scope='g_hr_up_h2')
				up_h2=tf.nn.relu(up_h2)
				up_h3=linear(up_h2,self.gf_dim*8,scope='g_hr_up_h3')
				up_h3=tf.nn.relu(up_h3)
				output=linear(up_h3,self.output_size**2*3,scope="g_hr_output")
				output=tf.reshape(output,[self.batch_size,self.output_size,self.output_size,3])
			
			return tf.nn.tanh(output)


	def discriminator_hr(self, image, y=None, reuse=False):
		with tf.variable_scope("discriminator_hr") as scope:
			if reuse:
				scope.reuse_variables()

			with tf.variable_scope("down_sampling") as scope:
				down_h0 = lrelu(conv2d(image, self.df_dim,k_h=4, k_w=4, name='d_hr_down_h0'))
				down_h1 = lrelu(conv2d(down_h0, self.df_dim * 2, k_h=4, k_w=4, name='d_hr_down_h1'))
				down_h2 = lrelu(conv2d(down_h1, self.df_dim * 4, k_h=4, k_w=4, name='d_hr_down_h2'))
				down_h3 = lrelu(conv2d(down_h2, self.df_dim * 8, k_h=4, k_w=4, name='d_hr_down_h3'))
				down_h4 = lrelu(conv2d(down_h3, self.df_dim * 16,k_h=4, k_w=4, name='d_hr_down_h4'))
				down_h5 = lrelu(conv2d(down_h4, self.df_dim * 32, k_h=4, k_w=4, name='d_hr_down_h5'))
				#size of down_h5: 4*4
				down_h6 = lrelu(conv2d(down_h5, self.df_dim * 16, k_h=1, k_w=1, 
								d_h=1, d_w=1,name='d_hr_down_h6'))
				#down_h7 = conv2d(down_h6, self.df_dim * 8, k_h=1, k_w=1,
				#				d_h=1, d_w=1, name='d_hr_down_h7')
				down_h7 = lrelu(conv2d(down_h6, self.df_dim * 8, k_h=1, k_w=1,
								d_h=1, d_w=1, name='d_hr_down_h7'))
				r3=down_h7
			'''
			#residual_block
			with tf.variable_scope("residual"):
				r0=lrelu(conv2d(down_h7,self.df_dim*2,k_h=1,k_w=1,d_h=1,d_w=1,name="d_hr_r0"))
				r1=lrelu(conv2d(r0,self.df_dim*2,k_h=3,k_w=3,d_h=1,d_w=1,name="d_hr_r1"))
				r2=conv2d(r1,self.gf_dim*8,k_h=3,k_w=3,d_h=1,d_w=1,name="d_hr_r2")
				r3=lrelu(tf.add(down_h7,r2))
			'''

			if self.y_dim:
				# When the spatial dimension is 4*4, replicate the description embedding spatially
				# perform a depth concatenation
				y_compressed = lrelu(linear(y, self.y_dim, 'd_hr_em_to_y'))
				yb = tf.reshape(y_compressed, [self.batch_size, 1, 1, self.y_dim])
				joint_img_text= conv_cond_concat(r3, yb) #4*4*128*9

			#4*4*128*8
			h4=lrelu(conv2d(joint_img_text,self.df_dim*8,k_h=1,k_w=1,d_h=1,d_w=1,name='d_hr_compress'))
			h5=conv2d(h4,1,k_h=4,k_w=4,d_h=1,d_w=1,name='d_hr_output')
			return tf.nn.sigmoid(h5), h5

	def sampler_hr(self, image_lr, y=None):
		with tf.variable_scope("generator_hr") as scope:
			scope.reuse_variables()

			#encode images
			with tf.variable_scope("down_sampling") as scope_0:
				image_lr_flatten=tf.reshape(image_lr,[self.batch_size,-1])
				#down_h0 64*64
				down_h0=linear(image_lr_flatten,self.gf_dim*8,scope="g_hr_down_h0")
				down_h0=tf.nn.relu(down_h0)
				#down_h1 32*32
				down_h1=linear(down_h0,self.gf_dim*8,scope="g_hr_down_h1")
				down_h1=tf.nn.relu(down_h1)
				#down_h2 16*16
				down_h2=linear(down_h1,self.gf_dim*8,scope="g_hr_down_h2")
				down_h2=tf.nn.relu(down_h2)

			#joint_img_text
			with tf.variable_scope("joint_img_text") as scope_1:
				y_compressed=tf.nn.relu(linear(y,self.y_dim,"g_hr_em_to_y_mean"))
				#yb = tf.reshape(y_compressed, [self.batch_size, 1, 1, self.y_dim])
				#joint_img_text= conv_cond_concat(down_h2, yb)
				# -->1batch_size * (128+2048)
				joint_img_text=concat([y_compressed,down_h2],1)
				# -->16 * 16 * 512
				#joint_img_text=tf.nn.relu(joint_img_text)
				r3=joint_img_text

			'''
			#residual block*4
			with tf.variable_scope("residuals") as scope_2:
				r0=self.residual_block(joint_img_text,name="g_hr_r0")
				r1=self.residual_block(r0,name="g_hr_r1")
				r2=self.residual_block(r1,name="g_hr_r2")
				r3=self.residual_block(r2,name="g_hr_r3")
			'''

			#upsampling
			with tf.variable_scope("upsampling") as scope_3:	
				up_h0=linear(r3,self.gf_dim*8,scope='g_hr_up_h0')
				up_h0=tf.nn.relu(up_h0)
				up_h1=linear(up_h0,self.gf_dim*8,scope='g_hr_up_h1')
				up_h1=tf.nn.relu(up_h1)
				up_h2=linear(up_h1,self.gf_dim*8,scope='g_hr_up_h2')
				up_h2=tf.nn.relu(up_h2)
				up_h3=linear(up_h2,self.gf_dim*8,scope='g_hr_up_h3')
				up_h3=tf.nn.relu(up_h3)
				output=linear(up_h3,self.output_size**2*3,scope="g_hr_output")
				output=tf.reshape(output,[self.batch_size,self.output_size,self.output_size,3])
			
			return tf.nn.tanh(output)

	def residual_block(self, x_c_code, name,train=True):
		with tf.variable_scope(name) as scope:
			h0=conv2d(x_c_code,self.gf_dim*4,k_h=3,k_w=3,d_h=1,d_w=1,name="conv0")
			h0=tf.contrib.layers.batch_norm(h0,decay=0.9, 
                                            updates_collections=None,
                                            epsilon=1e-5,
                                            scale=True,
                                            is_training=train,
                                            scope='bn0')
			h0=tf.nn.relu(h0)
			h1=conv2d(h0,self.gf_dim*4,k_h=3,k_w=3,d_h=1,d_w=1,name="conv1")
			h1=tf.contrib.layers.batch_norm(h1,decay=0.9, 
                                            updates_collections=None,
                                            epsilon=1e-5,
                                            scale=True,
                                            is_training=train,
                                            scope='bn1')
			h2=tf.nn.relu(tf.add(x_c_code,h1))
			return h2

	def train(self, config):
		"""Train DCGAN"""
		data = glob(os.path.join("./data", config.dataset, "*.jpg"))
		# np.random.shuffle(data)

		d_optim = tf.train.AdamOptimizer(self.learning_rate_d, beta1=config.beta1) \
			.minimize(self.d_loss, var_list=self.d_vars)
		g_optim = tf.train.AdamOptimizer(self.learning_rate_g, beta1=config.beta1) \
			.minimize(self.g_loss, var_list=self.g_vars)

		try:
			init_op = tf.global_variables_initializer()
		except:
			init_op = tf.initialize_all_variables()

		self.sess.run(init_op)

		if self.y_dim:
			if config.is_CA:
			   self.g_sum = merge_summary([self.g_loss_kl_sum,self.g_loss_im_sum,
									self.g_loss_sum])
			else:
				self.g_sum = merge_summary([self.g_loss_sum])
			self.d_sum = merge_summary([self.d_loss_distance_sum,
									self.d_loss_la_sum,
									self.gradient_penalty_sum,
									self.slopes_sum])
		else:
			self.g_sum = merge_summary([self.g_loss_sum])
			self.d_sum = merge_summary([self.d_loss_distance_sum,
									self.gradient_penalty_sum,self.slopes_sum])
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
			sample_ids = [int(x[31:-4]) for x in sample_files]
			sample_labels = [embedding_dict[x] for x in sample_ids]
			#sample_captions=[caption_dict[x] for x in sample_ids]

		if (self.is_grayscale):
			sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
		else:
			sample_images = np.array(sample).astype(np.float32)

		counter = 1
		start_time = time.time()

		if config.is_train:
			self.load_stage1()

		if self.load(self.checkpoint_dir):
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		generator_lr=config.learning_rate_g
		discriminator_lr=config.learning_rate_d
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
					
					batch_ids = [int(x[31:-4]) for x in batch_files]
					batch_labels = [embedding_dict[x] for x in batch_ids]

					batch_files_fake=data[(idx+1) * config.batch_size:(idx + 2) * config.batch_size]
					batch_ids_fake = [int(x[31:-4]) for x in batch_files_fake]
					batch_labels_fake=[embedding_dict[x] for x in batch_ids_fake]


				if (self.is_grayscale):
					batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
				else:
					batch_images = np.array(batch).astype(np.float32)

				batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
					.astype(np.float32)

				#learning rate decay
				if epoch % config.lr_decay_step == 0 and epoch != 0:
						generator_lr *= 0.5
						discriminator_lr *= 0.5

				# Update D network
				for iter in range(config.d_iters):
					if self.y_dim:
						_, summary_str = self.sess.run([d_optim, self.d_sum],
												   feed_dict={self.images: batch_images, self.z: batch_z,
															  self.embedding: batch_labels,
															  self.embedding_fake:batch_labels_fake,
															  self.learning_rate_d:discriminator_lr})
					else:
						_, summary_str = self.sess.run([d_optim, self.d_sum],
												   feed_dict={self.images: batch_images, self.z: batch_z,
																self.learning_rate_d:discriminator_lr})
					self.writer.add_summary(summary_str, counter)

				# Update G network
				if(counter>config.d_pre_train_step):
					for iter in range(config.g_iters):
						if self.y_dim:
							_, summary_str = self.sess.run([g_optim, self.g_sum],
											feed_dict={self.z: batch_z,self.embedding: batch_labels,
													   self.images: batch_images,
													   self.embedding_fake: batch_labels_fake,
													   self.learning_rate_g:generator_lr})
							self.writer.add_summary(summary_str, counter)


						else:
							_, summary_str = self.sess.run([g_optim, self.g_sum],
													   feed_dict={self.z: batch_z,
													   self.learning_rate_g:generator_lr})
							self.writer.add_summary(summary_str, counter)

				#Loss Evaluation
				if self.y_dim:
					errD=self.d_loss.eval({self.z: batch_z, self.embedding: batch_labels,
													   self.images: batch_images,
													   self.embedding_fake: batch_labels_fake})
					errG = self.g_loss.eval({self.z: batch_z, self.embedding: batch_labels})
				else:
					errD=self.d_loss.eval({self.z: batch_z, self.images: batch_images})
					errG = self.g_loss.eval({self.z: batch_z})

				counter += 1
				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
					% (epoch, idx, batch_idxs,
					time.time() - start_time, errD, errG))

				if np.mod(counter, 100) == 1:
					if self.y_dim:
						if self.is_CA:
							samples,_ = self.sess.run(self.sampler,
												feed_dict={self.z: sample_z,
															self.embedding: sample_labels})
						else:
							samples = self.sess.run(self.sampler,
												feed_dict={self.z: sample_z,
															self.embedding: sample_labels})
					else:
						samples = self.sess.run(self.sampler,
												feed_dict={self.z: sample_z})
					samples=samples[:16,:,:,:]
					save_images(samples, [4, 4],
						'./{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
					#print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

				if np.mod(counter, 500) == 2:
					self.save(config.checkpoint_dir, counter)		


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


	