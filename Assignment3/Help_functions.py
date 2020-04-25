import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import optimizers
from keras import backend as K


# This file constitutes a library of functions for reformatting and plotting etc.

def get_dense_conv_shape(encoder):
	for l in encoder.layers:
		if (len(l.input_shape) > len(l.output_shape)):
			return l.output_shape[1], l.input_shape[1:]

def get_data_predictions(autoencoder, n):
	x, x_pred = autoencoder.x_train[:n], autoencoder.model.predict(autoencoder.x_train[:n])
	return x, x_pred

def get_data_predictions_labels(autoencoder, n=None):
	model = autoencoder.model
	if n is None:
		n = autoencoder.x_train.shape[0]
	x, x_pred, label = autoencoder.x_train[:n], model.predict(autoencoder.x_train[:n]), autoencoder.y_train[:n]
	return x, x_pred, label

def nll(x_true, x_pred):
	""" Negative log likelihood (Bernoulli). """
	x_true, x_pred = K.reshape(x_true, (-1, 784)), K.reshape(x_pred, (-1, 784))
	return K.sum(K.binary_crossentropy(x_true, x_pred), axis=-1)

def reshape_img(img):
	if(img.shape[-1]==1):
		return img.reshape(img.shape[:-1])
	return img

def display_reconstructions(x,x_pred,n=16):
	# y_train = autoencoder.y_train[:n]
	if n > 8:
		plt.figure(figsize=(20, 6))
	else:
		plt.figure(figsize=(20, 10))
	plt.suptitle("Autoencoder reconstructions", fontsize=40)
	for i in range(n):
		# display original
		ax = plt.subplot(2, n, i + 1)
		plt.imshow(reshape_img(x[i]))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# display reconstruction
		ax = plt.subplot(2, n, i + 1 + n)
		plt.imshow(reshape_img(x_pred[i]))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()

def display_images(images,labels=None, n=16, title=None):
	#images = autoencoder.generate(n)
	images = images[:n]
	no_images = images.shape[0]
	channels = images.shape[-1]
	# Do the plotting
	plt.Figure()
	no_rows = np.ceil(np.sqrt(no_images))
	no_cols = np.ceil(no_images / no_rows)
	for img_idx in range(no_images):
		plt.subplot(no_rows, no_cols, img_idx + 1)
		if channels == 1:
			plt.imshow(images[img_idx, :, :, 0], cmap="binary")
		else:
			plt.imshow(images[img_idx, :, :, :].astype(np.float))
		plt.xticks([])
		plt.yticks([])
		if labels is not None:
			plt.title(f"Class is {str(labels[img_idx]).zfill(channels)}")

	plt.suptitle(title)
	# Show the thing ...
	plt.show()

def get_most_anomalous_images(data, model, n=16):
	df_loss = pd.DataFrame(columns=['index', 'loss'])
	i = 0
	for img in data:
		img = np.expand_dims(img, axis=0)
		loss = model.model.evaluate(img, img, verbose=0)
		df_loss = df_loss.append({'index': i, 'loss': loss}, ignore_index=True)
		i += 1

	df_loss = df_loss.sort_values(['loss'], ascending=False)
	df_loss = df_loss.astype({'index': 'int32'})
	idx = df_loss['index'].values[:n]
	return np.array([data[i] for i in idx])

def vae_get_anomalous(data, model, n=16):
	df_loss = pd.DataFrame(columns=['index', 'loss'])
	N = 10000
	x_gen = model.generate(n=N).astype('float32')
	i = 0
	for x in data:
		x_arr = np.repeat(x[np.newaxis,...], N, axis=0).astype('float32')
		loss = np.mean(nll(x_arr, x_gen))
		df_loss = df_loss.append({'index': i, 'loss': loss}, ignore_index=True)
		i += 1
		if(i % 1000 == 0):
			print('{}/{} encoding samples'.format(i,N))
	df_loss = df_loss.sort_values(['loss'], ascending=False)
	df_loss = df_loss.astype({'index': 'int32'})
	idx = df_loss['index'].values[:n]
	return np.array([data[i] for i in idx])


def set_optimizer(optimizer, learning_rate):
	if optimizer == "adadelta":
		optimizer = optimizers.Adadelta(learning_rate=learning_rate)
	elif optimizer == "adagrad":
		optimizer = optimizers.Adagrad(learning_rate=learning_rate)
	elif optimizer == "adam":
		optimizer = optimizers.Adam(learning_rate=learning_rate)
	elif optimizer == "adamax":
		optimizer = optimizers.Adamax(learning_rate=learning_rate)
	elif optimizer == "nadam":
		optimizer = optimizers.Nadam(learning_rate=learning_rate)
	elif optimizer == "rmsprop":
		optimizer = optimizers.RMSprop(learning_rate=learning_rate)
	elif optimizer == "sgd":
		optimizer = optimizers.SGD(learning_rate=learning_rate)

	return optimizer

