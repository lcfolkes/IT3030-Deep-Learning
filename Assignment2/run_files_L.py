import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
from Assignment2 import Main, Help_functions
from Assignment2.Autoencoder import Autoencoder
from Assignment2.Classifier import Classifier
from Assignment2.Encoder import Encoder
from Assignment2.Help_functions import calc_accuracy_classifier
from Assignment2.Preprocessing import Data
from Assignment2.Autoencoder import Autoencoder

data = Data('mnist')
data.describe()

encoder = Encoder(data.d1_x, 32)
Help_functions.tsne_plot(encoder,data,"T-SNE plot untrained encoder")

print('Supervised classifier: ')
classifier_supervised = Classifier(data.d2_x_train, data.d2_y_train, encoder)
calc_accuracy_classifier(classifier_supervised, data.d2_x_test, data.d2_y_test)
Help_functions.tsne_plot(classifier_supervised.encoder,data,"T-SNE plot supervised trained encoder")

print('Autoencoder: ')
autoencoder = Autoencoder(data.d1_x, encoder)
Help_functions.tsne_plot(autoencoder.encoder,data,"T-SNE plot unsupervised training (autoencoder)")
Help_functions.display_reconstructions(autoencoder)

print("Semi_supervised: ")
classifier_supervised = Classifier(data.d2_x_train, data.d2_y_train, autoencoder.encoder)
Help_functions.tsne_plot(classifier_supervised.encoder,data,"T-SNE plot unsupervised training (autoencoder)")


#result_with_auto = calc_accuracy_classifier(classifier_semi_supervised, data.d2_x_test, data.d2_y_test)

#autoencoder = Autoencoder(data.d1_x, encoder)
#Main.display_reconstructions(autoencoder)
#print("Supervised: ")
#result_semi_supervised = calc_accuracy_classifier(classifier_supervised, data.d2_x_test, data.d2_y_test)
