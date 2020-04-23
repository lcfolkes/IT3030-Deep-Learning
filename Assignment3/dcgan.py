import tensorflow as tf
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, BatchNormalization, LeakyReLU, Conv2DTranspose, Conv2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from Assignment3.stacked_mnist import StackedMNISTData, DataMode
import time
import datetime
from IPython import display

class DCGAN:
	def __init__(self, gen):

		model_name = 'dcgan'
		dir_name = os.path.join('./models', model_name)
		os.makedirs(dir_name, exist_ok=True)
		gen_name = gen.get_gen_name()
		self.file_name = os.path.join(dir_name, gen_name + ".h5")

		x_train, y_train = gen.get_full_data_set(training=True)
		self.train_dataset = self.__get_train_dataset(x_train)

		# input image dimensions
		img_shape = x_train.shape[1:]
		# noise dimension
		self.noise_dim = 100

		self.generator = self.__generator(output_shape=img_shape)
		self.discriminator = self.__discriminator(input_shape=img_shape)
		self.generator_loss = self.__generator_loss
		self.discriminator_loss = self.__discriminator_loss

		# This method returns a helper function to compute cross entropy loss
		self.cross_entropy = BinaryCrossentropy(from_logits=True)

		# Optimizers
		learning_rate = 1e-4
		self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
		self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

		checkpoint_dir = './training_checkpoints'
		os.makedirs(checkpoint_dir, exist_ok=True)
		self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
		self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
										 discriminator_optimizer=self.discriminator_optimizer,
										 generator=self.generator,
										 discriminator=self.discriminator)

		EPOCHS = 5
		num_examples_to_generate = 16
		self.seed = tf.random.normal(shape=[num_examples_to_generate, self.noise_dim])
		self.__train(self.train_dataset, epochs=EPOCHS)

	### PREPROCESS DATA AND CREATE BATCHES
	def __get_train_dataset(self, x_train):
		x_train = x_train.astype('float32')
		x_train = x_train * 2. - 1.  # Normalize the images to [-1, 1]
		self.BUFFER_SIZE = 60000
		self.BATCH_SIZE = 256
		dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(
			buffer_size=self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder=True)#.prefetch(1)
		return dataset


	### MODELS
	def __generator(self, output_shape):

		# inputs = Input(shape=(self.noise_dim,), name='z')
		# x = Dense(7 * 7 * 256, use_bias=False)(inputs)
		# x = BatchNormalization()(x)
		# x = LeakyReLU()(x)
		# x = Reshape((7, 7, 256))(x)
		# x = Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
		# x = BatchNormalization()(x)
		# x = LeakyReLU()(x)
		# x = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
		# x = BatchNormalization()(x)
		# x = LeakyReLU()(x)
		# generated = Conv2DTranspose(output_shape[-1], (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)
		# generator = Model(inputs=inputs, outputs=generated, name='generator')
		# generator.summary()
		# return generator

		model = Sequential()
		model.add(Dense(7 * 7 * 256, use_bias=False, input_shape=(self.noise_dim,)))
		model.add(BatchNormalization())
		model.add(LeakyReLU())

		model.add(Reshape((7, 7, 256)))
		assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

		model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
		assert model.output_shape == (None, 7, 7, 128)
		model.add(BatchNormalization())
		model.add(LeakyReLU())

		model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
		assert model.output_shape == (None, 14, 14, 64)
		model.add(BatchNormalization())
		model.add(LeakyReLU())

		model.add(Conv2DTranspose(output_shape[-1], (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
		assert model.output_shape == (None, 28, 28, output_shape[-1])
		return model

	def __discriminator(self, input_shape):

		# inputs = Input(shape=input_shape, name='discriminator_input')
		# x = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(inputs)
		# x = LeakyReLU()(x)
		# x = Dropout(0.3)(x)
		# x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
		# x = LeakyReLU()(x)
		# x = Dropout(0.3)(x)
		# x = Flatten()(x)
		# discriminated = Dense(1)(x)
		# discriminator = Model(inputs=inputs, outputs=discriminated)
		# discriminator.summary()
		# return discriminator

		model = Sequential()
		model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=input_shape))
		model.add(LeakyReLU())
		model.add(Dropout(0.3))

		model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
		model.add(LeakyReLU())
		model.add(Dropout(0.3))

		model.add(Flatten())
		model.add(Dense(1))
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
		fig = plt.figure(figsize=(4, 4))

		for i in range(predictions.shape[0]):
			plt.subplot(4, 4, i + 1)
			plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
			plt.axis('off')

		plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
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
		no_batches = np.floor_divide(self.BUFFER_SIZE, self.BATCH_SIZE)
		for epoch in range(epochs):
			print("[INFO] starting epoch {}/{}...".format(epoch + 1, epochs))
			sys.stdout.flush()

			start = time.time()
			i = 0
			for image_batch in dataset:
				if i == 0:
					print("image_batch {}/{}...".format(i + 1, no_batches))
				elif (i+1) % 10 == 0:
					print("image_batch {}/{}...".format(i+1, no_batches))
				self.__train_step(image_batch)
				i += 1

			# Produce images for the GIF as we go
			display.clear_output(wait=True)
			self.__generate_and_save_images(self.generator, epoch + 1, self.seed)

			# Save the model every 15 epochs
			#if (epoch + 1) % 15 == 0:
			self.checkpoint.save(file_prefix=self.checkpoint_prefix)

			print('Time for epoch {} is {}'.format(epoch + 1, datetime.timedelta(seconds=time.time() - start)))

		# Generate after the final epoch
		display.clear_output(wait=True)
		self.__generate_and_save_images(self.generator, epochs, self.seed)

if __name__ == "__main__":
	gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=9)
	dcgan = DCGAN(gen)


