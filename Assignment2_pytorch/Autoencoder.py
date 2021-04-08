from datetime import datetime
from Assignment2_pytorch import Help_functions
import torch
import torch.nn as nn
from Assignment2_pytorch.Preprocessing import Data
from Assignment2_pytorch.Encoder import Encoder
from Assignment2_pytorch.Decoder import Decoder


# This class combines an encoder model with a decoder model to create an autoencoder model

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, learning_rate=0.01, loss_function='binary_crossentropy', optimizer='adam',
                 epochs=20):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    # Dataset parameters
    dataset_name = 'mnist'
    dss_frac = 0.4
    dss_d1_frac = 0.6
    d2_train_frac = 0.5
    d2_val_frac = 0.7

    # Create and split dataset
    data = Data(dataset_name=dataset_name, dss_frac=dss_frac, dss_d1_frac=dss_d1_frac,
                d2_train_frac=d2_train_frac, d2_val_frac=d2_val_frac)

    # Print data summary
    #data.describe()
    encoder = Encoder(64)
    #encoded_data = encoder.forward(data.d2_x_test)
    decoder = Decoder(64)
    #decoded_data = decoder.forward(encoded_data, )
    autoencoder = Autoencoder(encoder, decoder)
    autoencoder.forward(data.d2_x_test)

