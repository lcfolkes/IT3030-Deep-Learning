from Assignment2 import Main
from Assignment2.Autoencoder import Autoencoder
from Assignment2.Classifier import Classifier
from Assignment2.Help_functions import calc_accuracy_classifier
from Assignment2.Decoder import Decoder
from Assignment2.Encoder import Encoder
from Assignment2.Preprocessing import Data
import matplotlib.pyplot as plt
from keras.metrics import categorical_accuracy
from keras.losses import categorical_crossentropy

data = Data('mnist')
encoder = Encoder(data.d1_x, 64).model

# classifier = Classifier(data.d2_x_train, data.d2_y_train, encoder).model
# accuracy = calc_accuracy_classifier(classifier, data.d2_x_test, data.d2_y_test)

autoencoder = Autoencoder(data.d1_x, encoder)
autoencoder.get_data_predictions()
