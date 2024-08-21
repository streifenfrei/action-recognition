import argparse
import os
import yaml

import wandb

import torch.nn.functional
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import ActionDataset
from models import create_model, create_optimizer

LOG_INTERVAL = 1000

name = None
config = None
model = None
optimizer = None
training_loader = None
validation_loader = None
loss_fn = None

def get_loss_fn(class_groups):
    def loss_fn(y_true, y_pred):
        groups_y_true = y_true.split(class_groups, -1)
        groups_y_pred = y_pred.split(class_groups, -1)
        loss = 0
        for i in range(len(groups_y_true)):
            loss += torch.nn.functional.cross_entropy(groups_y_true[i], groups_y_pred[i])
        return loss
    return loss_fn


def train_one_epoch():
    running_loss = 0.
    running_loss_epoch = 0

    for i, data in enumerate(training_loader):
        image = data["image"]
        sensor_data = data["sensor_data"]
        targets = data["targets"]

        optimizer.zero_grad()
        outputs = model(image)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_loss_epoch += loss.item()
        if i % LOG_INTERVAL == LOG_INTERVAL - 1:
            print(f"\r  batch {i + 1}, loss: {running_loss / LOG_INTERVAL} ({int((i+1) / len(training_loader) * 100)}%)", end="")
            running_loss = 0.
    return running_loss_epoch / len(training_loader)

def train(epoch_start, epoch_end):
    for epoch in range(epoch_start, epoch_end):
        print('\rEPOCH {}:'.format(epoch))
        model.train(True)
        training_loss = train_one_epoch()

        validation_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(validation_loader):
                image = data["image"]
                sensor_data = data["sensor_data"]
                targets = data["targets"]
                outputs = model(image)
                validation_loss += loss_fn(outputs, targets)
        validation_loss /= (i + 1)

        print(f"LOSS training: {training_loss} validation: {validation_loss}")
        wandb.log({"training loss": training_loss, "validation loss": validation_loss})
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        }, f"checkpoint.pth")

        artifact = wandb.Artifact(name=name, type="model")
        artifact.add_file(local_path="checkpoint.pth", name=name)
        artifact.save()
        os.remove("checkpoint.pth")


if __name__ == '__main__':
    # run the script like e.g.:
    # python ./main.py /path/to/dataset -r /eit_data_PCB4 -r /eit_data_PCB5
    argparser = argparse.ArgumentParser()
    argparser.add_argument("training_set")
    argparser.add_argument("validation_set")
    argparser.add_argument("-c", "--config", default="config.yaml")
    argparser.add_argument("-n", "--name", default="default")
    argparser.add_argument("-r", "--ros", action="append")
    args = argparser.parse_args()
    name = args.name
    config = yaml.safe_load(open(args.config))

    # create data loader
    training_dataset = ActionDataset(args.training_set, ros_topic_filter=args.ros, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    validation_dataset = ActionDataset(args.validation_set, ros_topic_filter=args.ros, annotation_map=training_dataset.annotation_map)
    training_loader = DataLoader(training_dataset, batch_size=config["batch_size"], shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=True)
    loss_fn = get_loss_fn(training_dataset.class_groups)

    # create model and optimizer, load checkpoint
    model = create_model(config["model"], training_dataset.target_size)
    optimizer = create_optimizer(config["optimizer"], model.parameters())
    wandb.init()
    try:
        epoch_start = 0
        try:
            checkpoint = torch.load(wandb.use_artifact(f"{name}:latest", type="model").download() + f"/{name}")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            epoch_start = checkpoint["epoch"] + 1
        except wandb.CommError:
            print("No checkpoint found. Starting training from scratch.")

        # train
        train(epoch_start, config["epochs"])
    except KeyboardInterrupt:
        pass
    finally:
        wandb.finish()