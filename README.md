# DCGANs
An Implementation of Deep Convolutional GANs using PyTorch and well structured using OOP.
this project was written on PC has the following:
- Windows 11
- Nvidia : GTX 1050 Ti
- CPU : i7-8750H
- RAM : 16 GB
- CUDA : 11.8 (from pytorch using pip command)
- Anaconda with Python 3.11+

Parameters:
before running the project make sure you have Nvidia CUDA installed on you PC.
Exaplanation :
CHANNELS_IMG = 3 (This is automatically uncomment, I assumed that MRI dataset or any other dataset is RGB image which means that you have 3 channels)
# CHANNELS_IMG = 1 (If you are sure you MNIST data or any data that has monocolor you can uncomment this line)
Z_DIM = 100 (the noise dimention)
NUM_EPOCHS = 50 (how manytimes the data will be training at once)


to run the script:
python ./Training.py

