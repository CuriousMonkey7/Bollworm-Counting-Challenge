import ml_collections as mlc
import torch
import os


def get_config(runlocal=False):
    config = mlc.ConfigDict()
    config.seed = 42
    config.debug = False
    config.run_name = "dropout0.3_efficientnet_b4_aug++"
    config.exp_name = "handle_white_neg"
    config.model_name = "efficientnet_b4"
    config.backbone_weights = "ImageNet"
    config.train_bs = 64
    config.valid_bs = config.train_bs * 2
    config.loss = ""
    config.img_size = (256, 256)
    
    config.loss_scale =0.1
    config.epochs = 50
    config.lr = 1e-4
    config.min_lr = 1e-6
    config.wd = 1e-3
    config.scheduler = "None"

    config.n_fold = 5
    config.num_classes = 2
    config.thr = 0.3

    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.num_workers = min(40, os.cpu_count())

    config.resume = ""
    if config.resume:
        config.wandb = dict()
        config.wandb.run_id = ""

    return config
