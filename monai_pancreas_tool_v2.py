import os
import tempfile

import torch
import nibabel as nib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from monai.data import Dataset, DataLoader
from monai.data.utils import pad_list_data_collate
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, 
                              ScaleIntensityRanged, CropForegroundd, Orientationd, 
                              Spacingd, ResizeD, ToTensord, AsDiscreted, 
                              ToDeviced, EnsureTypeD)


def get_transforms():
    """Define transformations."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTypeD(keys=['image']),
        ToDeviced(keys=["image"], device="cuda:2"), 
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1, 1, 1), mode=("bilinear")),
        ResizeD(keys=["image"], spatial_size=[224, 224, 144], mode=("trilinear")),
        ToTensord(keys=["image"]),
    ])

def get_orig_transforms():
    """Define transformations."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        Orientationd(keys=["image"], axcodes="RAS"),
       
    ])


def get_model(model_path="/home/model_codes/best_metric_model_diceceloss.pth"):
    """Load the trained model."""
    model = torch.load(model_path)
    model.eval()
    return model


def get_predictions(model, data_loader):
    """Run the model and return the original images and model predictions."""
    original_images = []
    model_predictions = []
    device = torch.device("cuda:2")

    with torch.no_grad():
        for data in data_loader:
            inputs = data["image"].to(device)
            outputs = model(inputs)
            outputs = np.squeeze(outputs, axis=0)
            original_images.append(inputs.cpu().numpy())
            model_predictions.append(outputs.cpu().numpy())
  
    return np.concatenate(original_images, axis=0), np.concatenate(model_predictions, axis=0)


def show_predictions(original_images, model_predictions, slice_idx, dimensions):
    """Display the original images and model predictions."""
    threshold = 0.5
    model_predictions = np.where(model_predictions >= threshold, 1, 0)

    blue_cmap = mcolors.LinearSegmentedColormap.from_list(
        'blue_alpha', [(0, 0, 1, i) for i in np.linspace(0, 1, 100)]
    )
    red_cmap = mcolors.LinearSegmentedColormap.from_list(
        'red_alpha', [(1, 0, 0, i) for i in np.linspace(0, 1, 100)]
    )

    orig_image = original_images[0, :, :, slice_idx]

    pancreas_class = model_predictions[0:1, :, :, :]
    tumor_class = model_predictions[1:2, :, :, :]
   

    pancreas_class_tensor = torch.from_numpy(pancreas_class)
    tumor_class_tensor = torch.from_numpy(tumor_class)
   
    resize_transform = ResizeD(keys=["pred"], spatial_size=dimensions, mode=('nearest'))
    resized_pancreas = resize_transform({"pred": pancreas_class_tensor})["pred"]
    resized_tumor = resize_transform({"pred": tumor_class_tensor})["pred"]

    plt.imshow(orig_image, cmap='gray')

    plt.imshow(resized_pancreas[ 0, :, :, slice_idx], cmap=blue_cmap, alpha=0.5)
    plt.imshow(resized_tumor[0, :, :, slice_idx], cmap=red_cmap, alpha=0.5)
    st.pyplot(plt)


def main():
    st.title('MONAI Model Prediction App')

    uploaded_file = st.file_uploader("Choose a file to upload", type=['nii', 'nii.gz'])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as f:
            f.write(uploaded_file.getvalue())
            temp_path = f.name

        data_dict = [{'image': temp_path}]

        #make the actual transformations
        transforms = get_transforms()
        data_set = Dataset(data=data_dict, transform=transforms)
        data_loader = DataLoader(data_set, batch_size=1, collate_fn=pad_list_data_collate)

        #get the original transformations
        orig_transforms=get_orig_transforms()
        orig_data_set = Dataset(data=data_dict, transform=orig_transforms)
        
        #get the original shapes so that we can convert it back to the original shape
        orig_example = orig_data_set[0]
        original_shapes=orig_example['image'].shape
        dimensions = list(original_shapes)[1:]

        slice_idx = orig_example['image'].shape[3] // 2 
        plt.imshow(orig_example['image'][0, :, :, slice_idx], cmap='gray')
        st.pyplot(plt)

        if st.button("Run Model"):
            _, model_predictions = get_predictions(get_model(), data_loader)
            show_predictions(orig_example['image'], model_predictions, slice_idx, dimensions)


if __name__ == "__main__":
    main()
