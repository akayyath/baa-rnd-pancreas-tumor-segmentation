from src.data import load_data, get_transforms, get_datasets_and_loaders
from src.model import segresnet
from src.train import get_loss_optimizer_scheduler, get_metrics_and_post_transformations, training_loop
import torch

def main():
    data_dir = "/home/Task07_Pancreas/"
    train_files, val_files = load_data(data_dir)
    train_transforms, val_transforms = get_transforms()
    train_loader, val_loader = get_datasets_and_loaders(train_files, val_files, train_transforms, val_transforms)
    
    in_channels = 1  
    out_channels = 2  
    max_epochs = 20
    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True

    model = segresnet(in_channels, out_channels)  
    device = torch.device('cuda:0')
    model = model.to(device)
    
    loss_function, optimizer, lr_scheduler = get_loss_optimizer_scheduler(model,device,max_epochs)
    dice_metric, dice_metric_batch, post_trans = get_metrics_and_post_transformations()

    
    training_loop(max_epochs, model, loss_function, optimizer, lr_scheduler, scaler, train_loader, val_loader, post_trans, dice_metric, dice_metric_batch, device)

if __name__ == "__main__":
    main()