import tensorflow as tf
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, BatchNormalization, LeakyReLU, Conv2DTranspose, Conv2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
import stacked_mnist
import time
import datetime
from IPython import display
import Help_functions


class DCGAN:
	def __init__(self, gen, epochs=30, force_learn=False):

		# Data
		x_train, y_train = gen.get_full_data_set(training=True)
		self.train_dataset = self.__get_train_dataset(x_train)

		# input image dimensions
		img_shape = x_train.shape[1:]
		# noise dimension
		self.noise_dim = 100

		num_examples_to_generate = 16
		self.seed = tf.random.normal(shape=[num_examples_to_generate, self.noise_dim])

		# initialize models
		self.generator = self.__generator(output_shape=img_shape)
		self.discriminator = self.__discriminator(input_shape=img_shape)
		self.generator_loss = self.__generator_loss
		self.discriminator_loss = self.__discriminator_loss

		# This method returns a helper function to compute cross entropy loss
		self.cross_entropy = BinaryCrossentropy(from_logits=True)

		# Optimizers
		learning_rate = 0.0002
		momentum = 0.5
		self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=momentum)
		self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=momentum)
		#self.generator_optimizer = tf.keras.optimizers.RMSprop()
		#self.discriminator_optimizer = tf.keras.optimizers.RMSprop()

		# Create checkpoints for training
		dir_name = './models/dcgan'
		os.makedirs(dir_name, exist_ok=True)
		self.gen_name = gen.get_gen_name()
		checkpoint_dir = os.path.join(dir_name, self.gen_name)
		os.makedirs(checkpoint_dir, exist_ok=True)

		self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
		self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
										 discriminator_optimizer=self.discriminator_optimizer,
										 generator=self.generator,
										 discriminator=self.discriminator)

		if force_learn:
			self.__train(self.train_dataset, epochs=epochs)
		else:
			try:
				status = self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
				status.assert_existing_objects_matched()
			except Exception as e:
				print(e)
				self.__train(self.train_dataset, epochs=epochs)

	### PREPROCESS DATA AND CREATE BATCHES
	def __get_train_dataset(self, x_train):
		x_train = x_train.astype('float32')
		x_train = x_train * 2. - 1.  # Normalize the images to [-1, 1]
		self.BUFFER_SIZE = 60000
		self.BATCH_SIZE = 256
		dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(
			buffer_size=self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder=True)
		return dataset


	### MODELS
	def __generator(self, output_shape):
		# ReLU and TanH activation according to paper

		model = Sequential()
		model.add(Dense(7 * 7 * 128, input_shape=(self.noise_dim,)))
		model.add(Reshape([7, 7, 128]))
		assert model.output_shape == (None, 7, 7, 128)

		model.add(BatchNormalization())
		# Selu for faster convergence
		model.add(Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='selu'))
		assert model.output_shape == (None, 14, 14, 64)

		model.add(BatchNormalization())
		model.add(Conv2DTranspose(output_shape[-1], kernel_size=(5, 5), strides=(2, 2), padding='same', activation='tanh'))
		assert model.output_shape == (None, 28, 28, output_shape[-1])

		return model


	def __discriminator(self, input_shape):
		# LeakyReLU with slope=0.2 according to paper

		model = Sequential()
		model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=input_shape))
		model.add(LeakyReLU(0.2))
		model.add(Dropout(0.3))

		model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
		model.add(LeakyReLU(0.2))
		model.add(Dropout(0.3))

		model.add(Flatten())
		model.add(Dense(1, activation='sigmoid'))
		return model


	### LOSS FUNCTIONS
	def __generator_loss(self, fake_output):
		return self.cross_entropy(tf.ones_like(fake_output), fake_output)

	def __discriminator_loss(self, real_output, fake_output):
		real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
		fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
		total_loss = real_loss + fake_loss
		return total_loss

	def __generate_and_save_images(self, model, epoch, test_input):

		predictions = model(test_input, training=False)
		channels = predictions.shape[-1]
		fig = plt.figure(figsize=(4, 4))

		for i in range(predictions.shape[0]):
			plt.subplot(4, 4, i + 1)
			if channels == 1:
				plt.imshow(predictions[i, :, :, 0], cmap="binary")
			else:
				plt.imshow((predictions[i, :, :, :].numpy()*255).astype(np.uint8))
			plt.axis('off')

		if epoch % 10 == 0:
			img_dir = os.path.join('./imgs/dcgan', self.gen_name)
			os.makedirs(img_dir, exist_ok=True)
			file_path = os.path.join(img_dir, 'image_at_epoch_{:04d}.png'.format(epoch))
			plt.savefig(file_path)
		plt.show()

	@tf.function
	def __train_step(self, images):
		noise = tf.random.normal(shape=[self.BATCH_SIZE, self.noise_dim])

		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			generated_images = self.generator(noise, training=True)

			real_output = self.discriminator(images, training=True)
			fake_output = self.discriminator(generated_images, training=True)

			gen_loss = self.generator_loss(fake_output)
			disc_loss = self.discriminator_loss(real_output, fake_output)

		gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
		gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

		self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
		self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

	def __train(self, dataset, epochs):
		for epoch in range(epochs):
			print("[INFO] starting epoch {}/{}...".format(epoch + 1, epochs))
			sys.stdout.flush()

			start = time.time()
			for image_batch in dataset:
				self.__train_step(image_batch)

			# Produce images for the GIF as we go
			display.clear_output(wait=True)
			self.__generate_and_save_images(self.generator, epoch + 1, self.seed)

			# Save the model every 10 epochs
			if (epoch + 1) % 10 == 0:
				self.checkpoint.save(file_prefix=self.checkpoint_prefix)

			print('Time for epoch {} is {}'.format(epoch + 1, datetime.timedelta(seconds=time.time() - start)))

		# Generate after the final epoch
		display.clear_output(wait=True)
		self.__generate_and_save_images(self.generator, epochs, self.seed)

	def generate(self, n=60000):
		noise = tf.random.normal(shape=[n, self.noise_dim])
		noise_dataset = tf.data.Dataset.from_tensor_slices(noise).batch(self.BATCH_SIZE)
		i = 0
		for noise_batch in noise_dataset:
			generated_img_batch = self.generator(noise_batch, training=False).numpy()
			if i == 0:
				generated_imgs = generated_img_batch
			else:
				generated_imgs = np.concatenate((generated_imgs, generated_img_batch), axis=0)
			i += 1
		return generated_imgs


if __name__ == "__main__":
	gen_standard = stacked_mnist.StackedMNISTData(mode=stacked_mnist.DataMode.MONO_FLOAT_COMPLETE, default_batch_size=2048)
	dcgan = DCGAN(gen_standard, force_learn=False)
	generated_imgs = dcgan.generate(n=60000)
	Help_functions.display_images(generated_imgs)


