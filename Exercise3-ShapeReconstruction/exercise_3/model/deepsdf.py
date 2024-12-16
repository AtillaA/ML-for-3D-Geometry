import torch.nn as nn
import torch


class DeepSDFDecoder(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.2

        # -----------------------------------------------------------------------
        # Define the model (DeepSDF auto-decoder architecture)
        self.lin0 = nn.utils.weight_norm(nn.Linear(latent_size + 3, 512))
        self.lin1 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.lin2 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.lin3 = nn.utils.weight_norm(nn.Linear(512, 512 - latent_size - 3))
        self.lin4 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.lin5 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.lin6 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.lin7 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.lin8 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        # -----------------------------------------------------------------------

    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        # -------------------------------------------
        # Forward pass implementation
        x = self.dropout(self.relu(self.lin0(x_in)))
        x = self.dropout(self.relu(self.lin1(x)))
        x = self.dropout(self.relu(self.lin2(x)))
        x = self.dropout(self.relu(self.lin3(x)))
        x = self.dropout(self.relu(self.lin4(torch.cat((x, x_in), 1))))
        x = self.dropout(self.relu(self.lin5(x)))
        x = self.dropout(self.relu(self.lin6(x)))
        x = self.dropout(self.relu(self.lin7(x)))
        x = self.lin8(x)
        # -------------------------------------------
        return x
