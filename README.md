# DCGANs

## Overview
This project is an implementation of Deep Convolutional Generative Adversarial Networks (DCGANs) using PyTorch, structured with Object-Oriented Programming (OOP) principles.

## System Specifications
- **Operating System:** Windows 11
- **GPU:** Nvidia GTX 1050 Ti
- **CPU:** Intel Core i7-8750H
- **RAM:** 16 GB
- **CUDA Version:** 11.8 (installed via PyTorch using pip command)
- **Anaconda:** Python 3.11+

## Parameters
Before running the project, ensure that Nvidia CUDA is installed on your PC.

### Explanation
- **CHANNELS_IMG:** 
    - Automatically uncommented: Assumes the dataset is RGB images with 3 channels.
    - Uncomment manually: If using MNIST dataset or any dataset with monocolor (1 channel).
- **Z_DIM:** Noise dimension (set to 100).
- **NUM_EPOCHS:** Number of training epochs (set to 50).

## MNIST Dataset
To use the MNIST dataset:
- Uncomment line 25 in `Training.py`.
- Comment line 24.

Note: The MNIST dataset is available to download via a Python script. Other datasets must be downloaded manually and placed in their respective folders (/MRI/, /celeb_dataset/, etc.).

## Edit the Transforms
- For 3 channels: Uncomment line 19 and comment line 20 in `Training.py`.
- For 1 channel: Uncomment line 20 and comment line 19 in `Training.py`.

### 3 Channels:
```python
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

### 1 Channel:
```python
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize((0.5,), (0.5,))
])
```

## Results
Here are two examples of generated images after training:

### Result 1
![MNIST](https://i.imgur.com/Qpy4seX.gif)

### Result 2
![MRI](https://i.imgur.com/GnDEDqP.gif)

## Test
You can test the project by running the `Test.py` file with the following command:
```bash
python Test.py
```
or
```bash
python main.py
```

## Run the Script
To run the script:
```bash
python Training.py
```
