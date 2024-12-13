import torch
import torch.nn as nn


class ThreeDEPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_features = 80

        # ------------------------------------------------------------------------------------------------------------------------------------
        # 4 Encoder layers
        self.enc1 = nn.Sequential(
            nn.Conv3d(2, self.num_features, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.enc2 = nn.Sequential(
            nn.Conv3d(self.num_features, self.num_features * 2, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(self.num_features * 2),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.enc3 = nn.Sequential(
            nn.Conv3d(self.num_features * 2, self.num_features * 4, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(self.num_features * 4),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.enc4 = nn.Sequential(
            nn.Conv3d(self.num_features * 4, self.num_features * 8, kernel_size=(4, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(self.num_features * 8),
            nn.LeakyReLU(negative_slope=0.2),
        )

        # 2 Bottleneck layers
        self.bottleneck = nn.Sequential(
            nn.Linear(self.num_features * 8, self.num_features * 8),
            nn.ReLU(),
            nn.Linear(self.num_features * 8, self.num_features * 8),
            nn.ReLU(),
        )

        # 4 Decoder layers
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(2 * self.num_features * 8, self.num_features * 4, kernel_size=(4, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(self.num_features * 4),
            nn.ReLU(),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(2 * self.num_features * 4, self.num_features * 2, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(2 * self.num_features),
            nn.ReLU(),
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(2 * self.num_features * 2, self.num_features, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(self.num_features),
            nn.ReLU(),
        )

        self.dec4 = nn.Sequential(
            nn.ConvTranspose3d(self.num_features * 2, 1, kernel_size=(4, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0)),
        )
        # ------------------------------------------------------------------------------------------------------------------------------------

    def forward(self, x):
        b = x.shape[0]

        # ---------------------------------------------------------------------------------------------
        # Encode (Pass x though encoder while keeping the intermediate outputs for the skip connections)
        x_e1 = self.enc1(x)
        x_e2 = self.enc2(x_e1)
        x_e3 = self.enc3(x_e2)
        x_e4 = self.enc4(x_e3)
        
        # Reshape and apply bottleneck layers
        x = x_e4.view(b, -1)
        x = self.bottleneck(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1, 1)

        # Decode (Pass x through the decoder, applying the skip connections in the process)
        x = self.dec1(torch.cat([x, x_e4], 1))
        x = self.dec2(torch.cat([x, x_e3], 1))
        x = self.dec3(torch.cat([x, x_e2], 1))
        x = self.dec4(torch.cat([x, x_e1], 1))

        # Log scaling
        x = torch.squeeze(x, dim=1)
        x = torch.log(torch.add(torch.abs(x), 1.0))
        # ---------------------------------------------------------------------------------------------

        return x
