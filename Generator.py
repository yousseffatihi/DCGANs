import torch
import torch.nn as nn

class Generator(nn.Module):
    """Some Information about Generator"""
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            # Input : N * z_dim * 1 * 1
            self._block(z_dim, features_g*16, 4, 1, 0), # N * f_g * 16 * 4 * 4
            self._block(features_g*16, features_g*8, 4, 2, 1), # 8 * 8
            self._block(features_g*8, features_g*4, 4, 2, 1), # 16 * 16
            self._block(features_g*4, features_g*2, 4, 2, 1), # 32 * 32
            nn.ConvTranspose2d(
                features_g*2, channels_img, kernel_size=4, stride=2, padding=1
            ), # Output : N * channels * 64 * 64
            nn.Tanh(), # Domain [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.generator(x)