from monai.handlers.utils import from_engine
from monai.inferers import sliding_window_inference
from monai.data.utils import decollate_batch
from monai.losses import DiceLoss, TverskyLoss,DiceCELoss
from monai.metrics import DiceMetric
import torch
import time
from monai.transforms import (Compose, Activations, AsDiscrete)

# Define Loss Function, Optimizer, and Scheduler
def get_loss_optimizer_scheduler(model,device,max_epochs):
    pancreas_weight=0.35
    tumor_weight=1.5
    ce_weights = torch.tensor([pancreas_weight, tumor_weight], device=device)
    loss_function = DiceCELoss(ce_weight=ce_weights, smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    return loss_function, optimizer, lr_scheduler

# Define Metrics and Post Transformations
def get_metrics_and_post_transformations():
    dice_metric = DiceMetric( reduction="mean")
    dice_metric_batch = DiceMetric( reduction="mean_batch")


    post_trans = Compose([
        Activations(sigmoid=True), 
        AsDiscrete(threshold=0.5)
    ])

    return dice_metric, dice_metric_batch, post_trans

# Inference Function

def inference(input,model):
    def _compute(input,model):
        with torch.no_grad():
            model.eval()
        return sliding_window_inference(
            inputs=input,
            roi_size=(224, 224, 144),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )
    VAL_AMP = True
    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input,model)
    else:
        return _compute(input,model)

    

# Training and Validation Loop
def training_loop(max_epochs, model, loss_function, optimizer, lr_scheduler, scaler, train_loader, val_loader, post_trans, dice_metric, dice_metric_batch, device):
    best_metric = -1  # Initialize the best metric value
    best_metric_epoch = -1  # Initialize the epoch corresponding to the best metric
    best_metrics_epochs_and_time = [[], [], []]  # List to store the best metric, epoch, and time
    epoch_loss_values = []  # List to store the loss value for each epoch
    metric_values = []  # List to store the metric value for each epoch
    metric_values_class1 = []  # List to store the metric value for class 1
    metric_values_class2 = []  # List to store the metric value for class 2
    model_dir = "/home/model_codes/config"  # Directory to save the best metric model
    val_interval = 1
    

    total_start = time.time()  # Start time for total training

    # Iterate over each epoch
    for epoch in range(max_epochs):
        epoch_start = time.time()  # Start time for the current epoch

        model.train()  # Set the model in training mode
        epoch_loss = 0  # Initialize the epoch loss
        step = 0  # Initialize the step count

        # Iterate over each batch in the training loader
        for batch_data in train_loader:
            step_start = time.time()  # Start time for the current step
            step += 1

            # Move inputs and labels to the specified device (GPU)
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

            optimizer.zero_grad()  # Clear the gradients
            with torch.cuda.amp.autocast():
                outputs = model(inputs)  # Forward pass
                loss = loss_function(outputs, labels)  # Compute the loss


            scaler.scale(loss).backward()  # Backward pass with gradient scaling
            scaler.step(optimizer)  # Update the model parameters
            scaler.update()  # Update the gradient scaler
            epoch_loss += loss.item()  # Accumulate the epoch loss

        lr_scheduler.step()  # Update the learning rate scheduler
        epoch_loss /= step  # Compute the average epoch loss
        epoch_loss_values.append(epoch_loss)  # Store the epoch loss
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Perform validation at specified intervals
        if (epoch + 1) % val_interval == 0:
            model.eval()  # Set the model in evaluation mode
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    val_outputs = inference(val_inputs,model)  # Perform inference on validation data
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    # val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)  # Compute the dice metric
                    dice_metric_batch(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()  # Compute the aggregate metric value
                metric_values.append(metric)  # Store the metric value
                metric_batch = dice_metric_batch.aggregate()
                metric_class1 = metric_batch[0].item()  # Compute the metric value for class 1
                metric_values_class1.append(metric_class1)  # Store the metric value for class 1
                metric_class2 = metric_batch[1].item()  # Compute the metric value for class 2
                metric_values_class2.append(metric_class2)  # Store the metric value for class 2
                
                dice_metric.reset()  # Reset the dice metric
                dice_metric_batch.reset()

                # Check if the current metric is better than the best metric
                if metric > best_metric:
                    best_metric = metric  # Update the best metric value
                    best_metric_epoch = epoch + 1  # Update the epoch corresponding to the best metric
                    best_metrics_epochs_and_time[0].append(best_metric)  # Store the best metric
                    best_metrics_epochs_and_time[1].append(best_metric_epoch)  # Store the corresponding epoch
                    best_metrics_epochs_and_time[2].append(time.time() - total_start)  # Store the total time
                    # torch.save(model, os.path.join(model_dir, "best_metric_model_diceceloss_v1.pth"))  # Save the best model
                    # print("Saved new best metric model")

                # Print the current and best metric values
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" class1: {metric_class1:.4f} class2: {metric_class2:.4f}"
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )

        print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")

    total_time = time.time() - total_start  # Compute the total training time
