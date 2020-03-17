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

encoder = Encoder(data.d1_x, 32)
#Help_functions.tsne_plot(encoder,data)
encoder_clean = Encoder(data.d1_x, 64)
#autoencoder = Autoencoder(data.d1_x, encoder.model)
#classifier_semi_supervised = Classifier(data.d2_x_train, data.d2_y_train, autoencoder.encoder)

classifier_supervised = Classifier(data.d2_x_train, data.d2_y_train, encoder_clean.model)

#Main.display_reconstructions(autoencoder)

print("Semi_supervised: ")
result_semi_supervised = calc_accuracy_classifier(classifier_semi_supervised, data.d2_x_test, data.d2_y_test)
print("Supervised: ")
result_supervised = calc_accuracy_classifier(classifier_supervised, data.d2_x_test, data.d2_y_test)
