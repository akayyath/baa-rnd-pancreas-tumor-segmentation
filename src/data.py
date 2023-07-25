import os
import glob
import torch
from monai.transforms import (
    Activations, AsDiscrete, Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    Transform, AsDiscreted, CropForegroundd,RandGaussianNoised,RandScaleIntensityd,
    NormalizeIntensityd, ResizeD,  ToTensord, ScaleIntensityRanged, Orientationd,
     Spacingd, ToDeviced,RandGaussianNoised,RandShiftIntensityd
)
from monai.data import DataLoader, Dataset, ArrayDataset, CacheDataset
from monai.data.utils import pad_list_data_collate, decollate_batch


# Define Paths and Load Data
def load_data(data_dir):
 
    train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

    # Create a list of dictionaries, each containing an image-label pair
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

    val_files, train_files = data_dicts[:2], data_dicts[25:27]

    return val_files, train_files

# Define Data Transformations
class GettheDesiredLabels(Transform):
    def __call__(self, data):
        # Extract the label from the input data
        label = data["label"]

        # Keep only the second and third channels from the label
        single_label = label[1:3]

        data["label"] = single_label

        return data

def get_transforms():
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),  # Load the image and label data from file
        EnsureChannelFirstd(keys=["image", "label"]),  # Ensure the channel dimension is the first dimension
        ToDeviced(keys=["image", "label"], device="cuda:0"),  # Move the data to the specified device (commented out)
        AsDiscreted(keys='label', to_onehot=3),  # Convert the label to one-hot encoding with 3 channels
        GettheDesiredLabels(),  # Extract the desired channels from the label
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),  # Rescale the intensity values of the image within the specified range
        CropForegroundd(keys=["image", "label"], source_key="image"),  # Crop the image and label to remove the background
        Orientationd(keys=["image", "label"], axcodes="RAS"),  # Adjust the image and label orientation to RAS
        Spacingd(keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")),  # Adjust the spacing of the image and label
        ResizeD(keys=["image", "label"], spatial_size=[224, 224, 144], mode=("trilinear", "nearest")),  # Resize the image and label to the specified spatial size
        NormalizeIntensityd(keys=["image"]),  # Normalize intensity values of the image
        RandScaleIntensityd(keys=["image"], factors=0.1),  # Apply random scaling of intensity values to the image
        RandShiftIntensityd(keys=["image"], offsets=0.1),  # Apply random shifting of intensity values to the image
        RandGaussianNoised(keys=["image"], std=0.1),  # Add random Gaussian noise to the image
        ToTensord(keys=["image", "label"]),  # Convert the image and label to PyTorch tensors
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),  # Load the image and label data from file
        EnsureChannelFirstd(keys=["image", "label"]),  # Ensure the channel dimension is the first dimension
        AsDiscreted(keys='label', to_onehot=3),  # Convert the label to one-hot encoding with 3 channels
        GettheDesiredLabels(),  # Extract the desired channels from the label
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),  # Rescale the intensity values of the image within the specified range
        CropForegroundd(keys=["image", "label"], source_key="image"),  # Crop the image and label to remove the background
        Orientationd(keys=["image", "label"], axcodes="RAS"),  # Adjust the image and label orientation to RAS
        Spacingd(keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")),  # Adjust the spacing of the image and label
        ResizeD(keys=["image", "label"], spatial_size=[224, 224, 144], mode=("trilinear", "nearest")),  # Resize the image and label to the specified spatial size
        ToTensord(keys=["image", "label"]),  # Convert the image and label to PyTorch tensors
    ])

    return train_transforms, val_transforms

# Create Dataset and DataLoader
def get_datasets_and_loaders(train_files, val_files, train_transforms, val_transforms):
    train_dataset = CacheDataset(data=train_files, transform=train_transforms)  # Create a dataset for training with the training files and specified transformations
    val_dataset = CacheDataset(data=val_files, transform=val_transforms)  # Create a dataset for validation with the validation files and specified transformations

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=pad_list_data_collate)  # Create a data loader for training with the training dataset
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=pad_list_data_collate)  # Create a data loader for validation wit

    return train_loader, val_loader