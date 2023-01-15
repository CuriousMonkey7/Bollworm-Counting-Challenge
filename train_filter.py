# python train.py -e baseline --run res18_wd
import random
import gc
import argparse
import os, shutil
import time
import copy
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.ops import sigmoid_focal_loss

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# WandB
import wandb

# Custom Import
from utils.loader import get_loaders
from utils.train_utils import train_one_epoch, valid_one_epoch
from configs import filter_config
from utils.models import build_model, load_model
from utils import secrets


def setup_wandb():
    api_key = secrets.wandb
    wandb.login(key=api_key)
    run = (
        wandb.init(
            project="bollworm",
            entity="curiousmonkey7",
            config={k: v for k, v in dict(vars(CFG)).items() if "__" not in k},
            name=CFG.run_name,
            group=CFG.exp_name,
        )
        if not CFG.resume
        else wandb.init(
            project="bollworm",
            entity="curiousmonkey7",
            config={k: v for k, v in dict(vars(CFG)).items() if "__" not in k},
            name=CFG.run_name,
            group=CFG.exp_name,
            id=CFG.wandb.run_id,
            resume="must",
            settings=wandb.Settings(start_method="thread"),
        )
    )
    return run


def set_seed(seed=42):
    # For reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"> SEEDING DONE {seed}")


def save_checkpoint(state, is_best, checkpoint, filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    print("best?", is_best)
    if is_best:
        # print("hmm")
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth.tar"))


def load_checkpoint(path, model, optimizer):
    print(f"Loading Checkpoint at {path}")
    checkpoint = torch.load(path)
    best = checkpoint["best_f1"], checkpoint["best_epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, best


def run_training(
    model, dataloaders, optimizer, scheduler, criterion, device, num_epochs
):
    # To automatically log gradients
    if not CFG.debug:
        wandb.watch(model, log_freq=100)

    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = -np.inf if not CFG.resume else CFG.best[0]
    best_epoch = -1 if not CFG.resume else CFG.best[1]
    history = defaultdict(list)

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        print(f"Epoch {epoch}/{num_epochs}")

        train_loss, train_scores = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device=CFG.device,
            epoch=epoch,
        )
        train_acc, train_f1 = train_scores

        val_loss, val_scores = valid_one_epoch(
            model, val_loader, criterion, device=CFG.device, epoch=epoch
        )
        val_acc, val_f1 = val_scores

        history["Train_Loss"].append(train_loss)
        history["Val_Loss"].append(val_loss)
        history["Train_Acc"].append(train_acc)
        history["Val_Acc"].append(val_acc)
        history["Train_f1"].append(train_f1)
        history["Val_f1"].append(val_f1)

        # Log the metrics
        lr = scheduler.get_last_lr()[0] if scheduler is not None else CFG.lr
        if not CFG.debug:
            wandb.log(
                {
                    "Train Loss": train_loss,
                    "Val Loss": val_loss,
                    "Train Acc": train_acc,
                    "Val Acc": val_acc,
                    "Train f1": train_f1,
                    "Val_f1": val_f1,
                }
            )

        print(
            f"Train F1:{train_f1} | Val F1: {val_f1:0.4f} | Train ACC: {train_acc:0.4f} | Val ACC: {val_acc:0.4f}"
        )

        # deep copy the model

        # last_model_wts = copy.deepcopy(model.state_dict())
        is_best = val_f1 > best_f1
        if is_best:
            print(f"Valid Score Improved ({best_f1:0.4f} ---> {val_f1:0.4f})")
            best_f1 = val_f1
            best_epoch = epoch
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "f1": val_f1,
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            checkpoint=CFG.run_path,
        )

        print()
        print()

    end = time.time()
    time_elapsed = end - start
    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600,
            (time_elapsed % 3600) // 60,
            (time_elapsed % 3600) % 60,
        )
    )
    print("Best F1 Score: {:.4f}".format(best_f1))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history


if __name__ == "__main__":
    # print("sleeping")
    # time.sleep(80*60)
    # print("ready")
    parser = argparse.ArgumentParser(description="Filter boolworm")

    # parser.add_argument(
    #     "-e", "--experiment", default="trail", type=str, help="enter experiment name"
    # )
    # parser.add_argument("--run", default="trail", type=str, help="enter run name")
    parser.add_argument("--debug", default=0, type=int, help="enter run name")
    parser.add_argument("--loss", default="", type=str, help="enter loss type")
    parser.add_argument("--viz", default=False, type=bool, help="")

    args = parser.parse_args()

    CFG = filter_config.get_config()
    # CFG.run_name = args.run
    # CFG.exp_name = args.experiment
    CFG.debug = bool(args.debug)
    CFG.loss = args.loss
    print(CFG.loss)

    CFG.run_path = f"runs/{CFG.exp_name}/{CFG.run_name}"
    os.makedirs(CFG.run_path, exist_ok=True)
    try:
        # Set Seed For Reproducibility
        set_seed(seed=CFG.seed)

        # Login WandB
        if not CFG.debug:
            run = setup_wandb()

        # Create Folds
        train_filter = pd.read_csv("data/preproc/train_filter.csv")
        fold =0
        train_df = train_filter.query("fold!=@fold").reset_index(drop=True)
        val_df = train_filter.query("fold==@fold").reset_index(drop=True)
        print(f"No of Examples used(train+val): {len(train_filter)}")
        # Get DataLoaders
        dataloaders = {"train":get_loaders(train_df,preload=True),"val":get_loaders(val_df,phase="val",preload=True)}
        if args.viz:
            trainiter = iter(dataloaders["train"])
            images,labels = next(trainiter)
            for i in range(5):
                plt.imshow(images[i].T)
                plt.savefig(f'{CFG.run_path}/{i}.jpg')   # save the fi
                plt.show()
        # Get Model
        model = build_model()
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")

        # Configure Optimizer, Scheduler, and Criterion
        optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
        scheduler = None  # lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        criterion =  nn.BCEWithLogitsLoss()  # nn.CrossEntropyLoss(weight=class_weights)

        if CFG.resume:
            model, optimizer, best = load_checkpoint(CFG.resume, model, optimizer)
            CFG.best = best

        # Run Training
        print("Starting Training")
        model, history = run_training(
            model,
            dataloaders,
            optimizer,
            scheduler,
            criterion,
            device=CFG.device,
            num_epochs=CFG.epochs if not CFG.debug else 2,
        )
        #  Save History
        torch.save(history, f"{CFG.run_path}/history.pth")
    except Exception as e:
        print(e)
    finally:
        if not CFG.debug:
            run.finish()
