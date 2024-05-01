import torch
import torch.nn as nn
from Discriminator import Discriminator
from Generator import Generator
from Initialization import init_weights

def Test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    discriminator = Discriminator(in_channels, 8)
    init_weights(discriminator)
    assert discriminator(x).shape == (N, 1, 1, 1)
    print("---\tINITIALIZATION OF DISCRIMINATOR SUCCESS\t---")
    generator = Generator(z_dim, in_channels, 8)
    init_weights(generator)
    z = torch.randn((N, z_dim, 1, 1))
    assert generator(z).shape == (N, in_channels, H, W)
    print("---\tINITIALIZATION OF GENERATOR SUCCESS\t---")
