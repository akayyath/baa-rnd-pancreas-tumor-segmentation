# Pancreatic Tumor Detection using MONAI

This project aims to detect and segment pancreatic tumors using the MONAI (Medical Open Network for AI) library, which provides deep learning tools for medical imaging. The code in this repository utilizes deep learning models and image processing techniques to automatically identify and localize pancreatic tumors in medical images.

## Table of Contents

- [Installation](#installation)
- [Data](#data)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Examples and Demonstrations](#examples-and-demonstrations)
- [Results and Discussion](#results-and-discussion)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact Information](#contact-information)

## Installation

To set up the project, follow these steps:

1. Connect to the server

2. Pull the image or use the existing image available in the server - ex:
  ```shell
  docker.io/monai/nvidia/cuda
  ```  


3. Spin up a new container using this image - ex: Run the following command: 
  ```shell
   docker run --name {name of the container} --gpus all -it --privileged -v /Downloads:/home --ipc=host { name of the image ex- monai/nvidia/cuda:11.8.0-ubuntu20.04}
  ```

  Replace <container_name> with a name for the container, <image_name> with the name of the Docker image (e.g., monai/nvidia/cuda), and /Downloads with the path to the local directory containing the code and data.


4. Clone the repository:

  ```shell
   git clone https://github.com/akayyath/baa-rnd-pancreas-tumor-segmentation.git
  ```

5. Install the necessary dependencies. Make sure you have Python and pip installed. 
```shell
pip install -r requirements.txt
```
This will install the required libraries, including MONAI, necessary for running the project.



The data is saved
home/Task07_Pancreas/
    |-- imagesTr/
    |-- imagesTs/
    |-- labelsTr/

## Model Training and Evaluation

The code provided in the repository includes the following components:

### Data Loading and Transformation

- The data loading and transformation steps are defined in the `get_transforms()` function and are applied to both training and validation datasets.
- Various MONAI transforms are used to preprocess the data, including rescaling intensity values, cropping, normalizing intensity, and augmenting the data with random shifts, scaling, and Gaussian noise.

### Model Architecture

- The model architecture used in this project is a SegResNet, which is a U-Net-like architecture with residual blocks.
- The model is defined in the `segresnet()` function and takes the number of input and output channels as arguments.

### Loss Function, Optimizer, and Scheduler

- The loss function used for training is a combination of Dice loss and Cross-Entropy loss (DiceCELoss).
- The optimizer is Adam with a learning rate of 1e-4 and weight decay of 1e-5.
- A CosineAnnealingLR scheduler is used to adjust the learning rate during training.

### Metrics and Post-Transformations

- The Dice metric is used to evaluate the model's performance during validation.
- The `post_trans` function applies sigmoid activation and converts the predicted probabilities to discrete labels.

### Training Loop

- The `training_loop()` function implements the main training and validation loop.
- The training loop iterates over the specified number of epochs, performs forward and backward passes, and updates the model's parameters.
- Validation is performed at specified intervals, and the best model based on the Dice metric is saved.
