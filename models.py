import torch
import torchvision


def create_model(config, out_dim):
    if config["type"] == 'resnet':
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Linear(2048, out_dim)
        return model
    else:
        ValueError("Unknown model type")

def create_optimizer(config, parameters):
    return torch.optim.Adam(parameters, lr=config["lr"])