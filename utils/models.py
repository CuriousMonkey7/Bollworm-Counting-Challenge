# Pytorch
import torch
import torch.nn as nn
from torchvision import models

# from torchvision.models import resnet50, ResNet50_Weights

# Custom Import
from configs import filter_config


CFG = filter_config.get_config()


def build_model():
    def get_fc(num_ftrs):
        return nn.Sequential(nn.Linear(num_ftrs, 128), 
                             nn.Dropout(p=0.5),
                             nn.Linear(128, 1))

    if CFG.model_name == "resnet18":
        print(f"Model Architecture: {CFG.model_name}")
        model_ft = models.resnet18(weights="IMAGENET1K_V1")
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = get_fc(num_ftrs)
        model_ft = model_ft.to(CFG.device)
        return model_ft

    if CFG.model_name == "resnet50":
        print(f"Model Architecture: {CFG.model_name}")
        model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = get_fc(num_ftrs)
        model_ft = model_ft.to(CFG.device)
        return model_ft
    
    elif CFG.model_name[:-3] == "efficientnet":
        print(f"Model Architecture: {CFG.model_name}")
        model_ft = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', f'nvidia_{CFG.model_name}', pretrained=True)
        num_ftrs = model_ft.classifier.fc.in_features
        model_ft.classifier.fc = get_fc(num_ftrs)
        model_ft = model_ft.to(CFG.device)
        return model_ft
        


def load_model(state_dict, inference=True):
    model = build_model()
    model.load_state_dict(state_dict)
    print(f"Loaded Model Stored")
    if inference:
        model.eval()
    return model
