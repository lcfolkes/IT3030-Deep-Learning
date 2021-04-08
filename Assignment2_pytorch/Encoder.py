import torch
import torch.nn as nn
from Assignment2_pytorch.Preprocessing import Data

print(torch.__version__)


# This class creates an encoder model

class Encoder(nn.Module):
    def __init__(self, size_latent_vector):
        super(Encoder, self).__init__()
        self.model = self.__encode(size_latent_vector)

    def __encode(self, size_latent_vector):
        model = nn.Sequential()
        # in: [b, 1, 26, 26]
        model.add_module("conv_1", nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3))  # [b, 16, 26, 26]
        model.add_module("relu_1", nn.ReLU()) # [b, 16, 26, 26]
        model.add_module("maxpool_1", nn.MaxPool2d(kernel_size=2, stride=2)) # [b, 16, 13, 13]

        model.add_module("conv_2", nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3)) # [b, 8, 11, 11]
        model.add_module("relu_2", nn.ReLU()) # [b, 8, 11, 11]
        model.add_module("maxpool_2", nn.MaxPool2d(kernel_size=2, stride=2)) # [b, 8, 5, 5]
        model.add_module("dropout", nn.Dropout(0.2)) # [b, 8, 5, 5]

        model.add_module("dense", nn.Linear(in_features=8*5*5, out_features=size_latent_vector)) # [b, 64]
        model.add_module("relu_3", nn.ReLU()) # [b, 64]

        return model

    def forward(self, x):
        print(x.size())
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                # Reshapes from [10,8,5,5] to [10, 200]. 10 is the batch size
                x = layer(x.view(x.size(0), -1))
                # When linear layer is applied, we get shape [10, size_latent_vector] = [10, 64]
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
    encoder = Encoder(64)
    encoder.forward(data.d2_x_test)