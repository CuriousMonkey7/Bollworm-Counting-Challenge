from tqdm import tqdm
import gc

# Pytorch
import torch
import torch.nn as nn
from torch.cuda import amp

# Custom Import
from configs import filter_config
from utils.metric import get_metrics

CFG = filter_config.get_config()


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, epoch):
    model.train()

    dataset_size = 0
    running_loss = 0.0
    train_data = {
        "preds": torch.empty(0).to(device),
        "labels": torch.empty(0).to(device),
    }

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Train")
    for step, (images, labels) in pbar:
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32).unsqueeze(1)

        batch_size = images.size(0)

        with torch.set_grad_enabled(True):
            y_pred = model(images)
            loss = criterion(y_pred, labels)

        loss.backward()
        optimizer.step()

        # zero the parameter gradients
        optimizer.zero_grad()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            train_loss=f"{epoch_loss:0.4f}",
            lr=f"{current_lr:0.5f}",
            gpu_mem=f"{mem:0.2f} GB",
        )

        y_pred = nn.Sigmoid()(y_pred)
        # y_pred = torch.argmax(y_pred, dim=1)
        y_pred[y_pred >= CFG.thr] = 1
        y_pred[y_pred < CFG.thr] = 0
        train_data["preds"] = torch.cat((train_data["preds"], y_pred))
        train_data["labels"] = torch.cat((train_data["labels"], labels))

    if scheduler is not None:
        scheduler.step()
    train_scores = get_metrics(train_data["preds"].int(), train_data["labels"].int())
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss, train_scores


@torch.no_grad()
def valid_one_epoch(model, dataloader, criterion, device, epoch, infer=False):
    model.eval()

    dataset_size = 0
    running_loss = 0.0
    val_data = {"preds": torch.empty(0).to(device), "labels": torch.empty(0).to(device)}

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Valid ")
    for step, (images, labels) in pbar:
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32).unsqueeze(1)

        batch_size = images.size(0)

        y_pred = model(images)
        loss = criterion(y_pred, labels)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size

        y_pred = nn.Sigmoid()(y_pred)
        # y_pred = torch.argmax(y_pred, dim=1)
        y_pred[y_pred >= CFG.thr] = 1
        y_pred[y_pred < CFG.thr] = 0
        val_data["preds"] = torch.cat((val_data["preds"], y_pred))
        val_data["labels"] = torch.cat((val_data["labels"], labels))

        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0

        pbar.set_postfix(valid_loss=f"{epoch_loss:0.4f}", gpu_memory=f"{mem:0.2f} GB")

    val_scores = get_metrics(val_data["preds"].int(), val_data["labels"].int())
    torch.cuda.empty_cache()
    gc.collect()
    if infer:
        return epoch_loss, val_scores, val_data
    return epoch_loss, val_scores



@torch.no_grad()
def test(model, dataloader, device):
    model.eval()
    val_data = {"preds": torch.empty(0).to(device)}

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Test ")
    for step, (images) in pbar:
        images = images.to(device, dtype=torch.float32)
        y_pred = model(images)
        y_pred = nn.Sigmoid()(y_pred)
        y_pred[y_pred >= CFG.thr] = 1
        y_pred[y_pred < CFG.thr] = 0
        val_data["preds"] = torch.cat((val_data["preds"], y_pred))

    torch.cuda.empty_cache()
    gc.collect()
    return val_data
