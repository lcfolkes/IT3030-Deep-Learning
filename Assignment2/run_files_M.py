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
#encoder_clean = Encoder(data.d2_x_train, 64)
print('Supervised classifier: ')
classifier_supervised = Classifier(data.d2_x_train, data.d2_y_train, encoder)
calc_accuracy_classifier(classifier_supervised, data.d2_x_test, data.d2_y_test)
Help_functions.tsne_plot(classifier_supervised.encoder,data,"T-SNE plot supervised trained encoder")
print("Semi_supervised: ")
#result_with_auto = calc_accuracy_classifier(classifier_semi_supervised, data.d2_x_test, data.d2_y_test)

#Main.display_reconstructions(autoencoder)

print("Semi_supervised: ")
result_semi_supervised = calc_accuracy_classifier(classifier_semi_supervised, data.d2_x_test, data.d2_y_test)
print("Supervised: ")
result_supervised = calc_accuracy_classifier(classifier_supervised, data.d2_x_test, data.d2_y_test)
