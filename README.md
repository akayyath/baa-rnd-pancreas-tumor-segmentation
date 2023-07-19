# baa-rnd-pancreas-tumor-segmentation
# Pancreatic Tumor Detection using MONAI

This project aims to detect pancreatic tumors using the MONAI library, which provides deep learning tools for medical imaging. The code in this repository utilizes deep learning models and image processing techniques to automatically identify and localize pancreatic tumors in medical images.

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
home/Task07_Pancreas
├── imagesTr/
├── imagesTs/
└── labelsTr/
