import torch
import torch.nn as nn

from Assignment2_pytorch.Encoder import Encoder
from Assignment2_pytorch.Preprocessing import Data

print(torch.__version__)

# This class creates a decoder model

class Decoder(nn.Module):
    def __init__(self, size_latent_vector):#, output_shape):
        super(Decoder, self).__init__()
        self.model = self.__decode(size_latent_vector)

    def __decode(self, size_latent_vector):
        """
        :param self:
        """
        model = nn.Sequential()
        # in: [b, 64]
        model.add_module("dense", nn.Linear(in_features=size_latent_vector, out_features=8*5*5)) # [b, 200]
        model.add_module("dropout", nn.Dropout(0.2)) # [b, 200]
        model.add_module("conv_t_1", nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2)) # [b, 8, 11, 11]
        model.add_module("relu_1", nn.ReLU()) # [b, 8, 11, 11]

        model.add_module("conv_t_2", nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, stride=1)) # [b, 16, 13, 13]
        model.add_module("relu_2", nn.ReLU()) # [b, 6, 13, 13]

        model.add_module("conv_t_3", nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, output_padding=1)) # [b, 1, 28, 28]
        model.add_module("sigmoid", nn.Sigmoid()) # [b, 1, 28, 28]
        return model

    def forward(self, x):
        for name, layer in self.model.named_modules():
            layer.auto_name = name
        print(x.size())
        for layer in self.model:
            if layer.auto_name == "conv_t_1":
                x = layer(x.view(x.size(0), 8, 5, 5))
            else:
                x = layer(x)
            print(x.size())

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
    print("Encoder")
    encoder = Encoder(64)
    encoded_data = encoder.forward(data.d2_x_test)
    print("Decoder")
    decoder = Decoder(64)
    decoded_data = decoder.forward(encoded_data, )



