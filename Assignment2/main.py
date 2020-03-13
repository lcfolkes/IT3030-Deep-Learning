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
		encoder = Encoder(data, size_latent_vector)
		autoencoder = Autoencoder(encoder, learning_rate_autoencoder, loss_function_autoencoder, optimizer_autoencoder,
								  epochs_autoencoder)

		classifier_pretrained = Classifier(autoencoder.encoder, learning_rate_classifier, loss_function_classifier,
										   optimizer_classifier, epochs_classifier, size_latent_vector,freeze)

		classifier = Classifier(encoder, learning_rate_classifier, loss_function_classifier, optimizer_classifier,
								epochs_classifier, size_latent_vector, freeze)

		self.__display_reconstructions(no_reconstructions)


	def __display_reconstructions(self, no):
		#display given number or reconstructions
		pass