import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from torch.utils.tensorboard import SummaryWriter
from Discriminator import Discriminator
from Generator import Generator
from Initialization import init_weights
from HyperParameters import device, LEARNING_RATE, BATCH_SIZE, IMAGE_SIZE, CHANNELS_IMG, Z_DIM, NUM_EPOCHS, FEATURES_OF_DISCRIMINATOR, FEATURES_OF_GENERATOR

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize((0.5,), (0.5,))
])

#Brain Tumor MRI Dataset
dataset = datasets.ImageFolder(root='./MRI/', transform=transform)
# dataset = datasets.MNIST(root='./dataset/', train=True, transform=transform, download=True)
# dataset = datasets.ImageFolder(root="./celeb_dataset/", transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
generator = Generator(Z_DIM, CHANNELS_IMG, FEATURES_OF_GENERATOR).to(device)
discriminator = Discriminator(CHANNELS_IMG, FEATURES_OF_DISCRIMINATOR).to(device)
init_weights(generator)
init_weights(discriminator)

generator_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

criterion = nn.BCELoss()

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

real_writer = SummaryWriter(f"logs/real")
fake_writer = SummaryWriter(f"logs/fake")

generator.train()
discriminator.train()

step = 0

# Define the directory to save the images
save_dir_fake = "./fake_images/"
save_dir_real = "./real_images/"

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
        fake = generator(noise)

        ### TRAIN DISCRIMINATOR max log(D(x)) + log(1 - D(G(z)))
        real_discriminator = discriminator(real).reshape(-1)
        loss_of_real_discriminator = criterion(real_discriminator, torch.ones_like(real_discriminator))
        fake_discriminator = discriminator(fake.detach()).reshape(-1)
        loss_of_fake_discriminator = criterion(fake_discriminator, torch.zeros_like(fake_discriminator))
        loss_of_discriminator = (loss_of_real_discriminator + loss_of_fake_discriminator) / 2
        discriminator.zero_grad()
        loss_of_discriminator.backward()
        discriminator_optimizer.step()

        ### TRAIN GENERATOR min log(1 - D(G(z))) <--> max log(D(G(z))) 
        output = discriminator(fake).reshape(-1)
        loss_of_generator = criterion(output, torch.ones_like(output))
        generator.zero_grad()
        loss_of_generator.backward()
        generator_optimizer.step()

        ### PRINT LOSSES
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_of_discriminator:.4f}, loss G: {loss_of_generator:.4f}"
            )

            with torch.no_grad():
                fake = generator(fixed_noise)

                ### TAKE OUT (UP TO) 32 EXAMPLES
                real_grid_images = torchvision.utils.make_grid(real[:32], normalize=True)
                fake_grid_images = torchvision.utils.make_grid(fake[:32], normalize=True)

                real_writer.add_image("Real", real_grid_images, global_step=step)
                fake_writer.add_image("Fake", fake_grid_images, global_step=step)

                os.makedirs(save_dir_fake, exist_ok=True)
                save_image(fake_grid_images, os.path.join(save_dir_fake, f"fake_images_{step}.png"))

                os.makedirs(save_dir_real, exist_ok=True)
                save_image(real_grid_images, os.path.join(save_dir_real, f"real_images_{step}.png"))
            
            step += 1