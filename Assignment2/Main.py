import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

from Assignment2.Classifier import Classifier
from Assignment2.Preprocessing import Data
from Assignment2.Autoencoder import Autoencoder
from Assignment2.Encoder import Encoder
from Assignment2.Decoder import Decoder




class Main:

	def __init__(self, dataset, learning_rate_autoencoder, learning_rate_classifier, loss_function_autoencoder,
				 loss_function_classifier, optimizer_autoencoder, optimizer_classifier, size_latent_vector,
				 epochs_autoencoder, epochs_classifier, dss_split,d2_train_frac,d2_val_frac, freeze,
				 no_reconstructions, tSNE):

		#D1 = unlabeled, D2 = labeled
		data = Data(dataset, dss_split, d2_train_frac,d2_val_frac)
		encoder = Encoder(data.d1_x, size_latent_vector)
		autoencoder = Autoencoder(data.d1_x, size_latent_vector, learning_rate_autoencoder, loss_function_autoencoder, optimizer_autoencoder,
								  epochs_autoencoder)
		display_reconstructions(autoencoder)
		classifier_pretrained = Classifier(autoencoder.encoder, learning_rate_classifier, loss_function_classifier,
										   optimizer_classifier, epochs_classifier, size_latent_vector,freeze)

		classifier = Classifier(encoder, learning_rate_classifier, loss_function_classifier, optimizer_classifier,
								epochs_classifier, size_latent_vector, freeze)


def display_reconstructions(autoencoder,n=16):
	x_test, decoded_imgs = autoencoder.get_data_predictions(16)
	plt.figure(figsize=(20, 4))
	for i in range(n):
		# display original
		ax = plt.subplot(2, n, i + 1)
		plt.imshow(reshape_img(x_test[i]))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# display reconstruction
		ax = plt.subplot(2, n, i + 1 + n)
		plt.imshow(reshape_img(decoded_imgs[i]))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()

def reshape_img(img):
	return img.reshape(img.shape[:-1])
