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
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

data = Data('cifar10')
#data.describe()

encoder1 = Encoder(data.d1_x, 32)
encoder2 = Encoder(data.d1_x, 32)

print('Supervised classifier: ')
Help_functions.tsne_plot(encoder1, data, "T-SNE plot untrained encoder1")
classifier_supervised = Classifier(data, encoder1)
d2_acc_super = calc_accuracy_classifier(classifier_supervised, data.d2_x_test, data.d2_y_test)
print(d2_acc_super)
Help_functions.tsne_plot(classifier_supervised.encoder,data,"T-SNE plot supervised trained encoder")

print('Autoencoder: ')
Help_functions.tsne_plot(encoder2, data, "T-SNE plot untrained encoder2")
autoencoder = Autoencoder(data, encoder2)
Help_functions.tsne_plot(autoencoder.encoder, data, "T-SNE plot unsupervised training (autoencoder)")
Help_functions.display_reconstructions(autoencoder)

print("Semi_supervised: ")
classifier_semi_supervised = Classifier(data, autoencoder.encoder)
d2_acc_semi_super = calc_accuracy_classifier(classifier_semi_supervised, data.d2_x_test, data.d2_y_test)
print(d2_acc_semi_super)
Help_functions.tsne_plot(classifier_semi_supervised.encoder,data,"T-SNE plot semi-supervised training")

print("Results D2 test data")
print("Supervised classifier")
d2_acc_super = calc_accuracy_classifier(classifier_supervised, data.d2_x_test, data.d2_y_test)
print("Semi-supervised classifier")
calc_accuracy_classifier(classifier_semi_supervised, data.d2_x_test, data.d2_y_test)

print("Results complete D1 data")
print("Supervised classifier")
calc_accuracy_classifier(classifier_supervised, data.d1_x, data.d1_y)
print("Semi-supervised classifier")
calc_accuracy_classifier(classifier_semi_supervised, data.d1_x, data.d1_y)

