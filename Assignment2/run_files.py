from Assignment2 import Main
from Assignment2.Autoencoder import Autoencoder
from Assignment2.Classifier import Classifier2
from Assignment2.Help_functions import calc_accuracy
from Assignment2.Decoder import Decoder
from Assignment2.Encoder import Encoder, modify_input_shape
from Assignment2.Preprocessing import Data
import matplotlib.pyplot as plt
from keras.metrics import categorical_accuracy
from keras.losses import categorical_crossentropy


a = Data('mnist')

#a = Autoencoder(data.d1_x, 64)
#Main.display_reconstructions(a)


b = Encoder(a, 32)
c = Classifier2(a.d2_x_train, a.d2_y_train, b)
#d = modify_input_shape(a.d2_x_test)
#print(d.shape)
e = calc_accuracy(c, a.d2_x_test, a.d2_y_test)
#f = categorical_accuracy(a.d2_y_test, e)
#g = categorical_crossentropy(a.d2_y_test, e)
#print(f[1])