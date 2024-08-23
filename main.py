import argparse
import os
import yaml

import wandb

import torch.nn.functional
from torchvision import transforms

from data_loader import load_annotation_map, load_data
from models import create_model, create_optimizer

LOG_INTERVAL = 100

name = None
config = None
model = None
optimizer = None
training_loader = None
validation_loader = None
loss_fn = None


def get_loss_fn(annotation_map):
    class_groups = []
    for track in annotation_map:
        for attribute in annotation_map[track]:
            class_groups.append(len(annotation_map[track][attribute]))

    def loss_fn(y_true, y_pred):
        groups_y_true = y_true.split(class_groups, -1)
        groups_y_pred = y_pred.split(class_groups, -1)
        loss = 0
        for i in range(len(groups_y_true)):
            loss += torch.nn.functional.cross_entropy(groups_y_true[i], groups_y_pred[i])
        return loss

    return loss_fn


def train_one_epoch():
    running_loss_epoch = 0
    i = 0
    batch_size = None
    total_length = sum(len(x) for x in training_loader)
    for loader in training_loader:
        running_loss = 0.
        for data in loader:
            image = data["image"]
            batch_size = image.shape[0]
            sensor_data = data["sensor_data"]   # not used yet
            targets = data["targets"]

            optimizer.zero_grad()
            outputs = model(image)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_loss_epoch += loss.item()
            if i % LOG_INTERVAL == LOG_INTERVAL - 1:
                print(
                    f"\r  batch {i + 1}, loss: {running_loss / LOG_INTERVAL} ({int((i + 1) * batch_size / total_length * 100)}%)",
                    end="")
                running_loss = 0.
            i += 1
    return running_loss_epoch * batch_size / total_length


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
    argparser.add_argument("config", default="config.yaml")
    argparser.add_argument("-n", "--name", default="default")
    args = argparser.parse_args()
    name = args.name
    config = yaml.safe_load(open(args.config))

    # create data loader
    data_config = config["data"]
    annotation_map = load_annotation_map(data_config["specfile"])
    ros_topics = config["ros_topics"] if "ros_topics" in config else None
    no_temporal_relation = config["model"]["type"] in ["resnet"]
    training_loader = load_data(data_config["training_set"], annotation_map, batch_size=config["batch_size"],
                                ros_topics=ros_topics, shuffle=no_temporal_relation, concat=no_temporal_relation,
                                transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ]))
    validation_loader = load_data(data_config["validation_set"], annotation_map, batch_size=config["batch_size"],
                                  ros_topics=ros_topics, shuffle=no_temporal_relation, concat=no_temporal_relation)
    loss_fn = get_loss_fn(annotation_map)

    # create model and optimizer, load checkpoint
    model = create_model(config["model"], sum(sum(len(y) for y in x.values()) for x in annotation_map.values()))
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
