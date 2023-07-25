from monai.networks.nets import SegResNet

# Define Model
def segresnet(in_channels, out_channels):
    # Create a SegResNet model with the specified parameters
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],  # Number of residual blocks in each downsampling path
        blocks_up=[1, 1, 1],  # Number of residual blocks in each upsampling path
        init_filters=16,  # Number of filters in the first layer
        in_channels=in_channels,  # Number of input channels
        out_channels=out_channels,  # Number of output channels
        dropout_prob=0.2,  # Dropout probability
    )

    return model

