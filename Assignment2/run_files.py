from Assignment2 import Main
from Assignment2.Autoencoder import Autoencoder
from Assignment2.Decoder import Decoder
from Assignment2.Encoder import Encoder
from Assignment2.Preprocessing import Data
import matplotlib.pyplot as plt

data = Data('mnist')
a = Autoencoder(data.d1_x, 64)
Main.display_reconstructions(a)
