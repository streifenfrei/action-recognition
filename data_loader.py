import numpy as np
import torch
import os

from collections import OrderedDict
import torchvision.io
from torch.utils.data import Dataset
from pathlib import Path
from lxml import etree
from rosbags.highlevel import AnyReader

VIDEO_FPS = 25


def ros_message_to_tensor(message, message_type):
    if message_type == "rokoko_ros/msg/Suit":
        message = torch.tensor(message.frame.data, device='cpu')
    elif message_type == "std_msgs/msg/String":
        message = torch.tensor([float(i) for i in str.split(message.data, sep=',') if
                                i.isnumeric() or i.replace(".", "", 1).isnumeric()], device='cpu')
    elif message_type == "sensor_msgs/msg/JointState":
        return torch.tensor(np.array([message.position, message.velocity]), device='cpu')
    # else:
    #    raise ValueError(f"Unknown ROS message type {message_type}")
    return message


def load_rosbag(path, filter=None):
    # load messages from ROS bag
    unaligned_data = {}
    with AnyReader([Path(path)]) as reader:
        seen_topics = []
        for i, (connection, timestamp, rawdata) in enumerate(reader.messages()):
            if filter is not None and connection.topic not in filter:
                continue
            if connection.topic not in seen_topics:
                seen_topics.append(connection.topic)
            if connection.topic not in unaligned_data:
                unaligned_data[connection.topic] = {"timestamps": [], "data": []}
            message = reader.deserialize(rawdata, connection.msgtype)
            unaligned_data[connection.topic]["timestamps"].append((timestamp - reader.start_time) * 1e-9)
            unaligned_data[connection.topic]["data"].append(ros_message_to_tensor(message, connection.msgtype))
    # align data points to the video frames
    aligned_data = {}
    # TODO: do actual interpolation here?
    for key in unaligned_data.keys():
        messages = iter(unaligned_data[key]["data"])
        message = next(messages)
        timestamps = iter(unaligned_data[key]["timestamps"])
        next(timestamps)
        timestamp = next(timestamps)
        aligned_data[key] = []
        for i in range(int(unaligned_data[key]["timestamps"][-1]) * VIDEO_FPS):
            while timestamp < i / VIDEO_FPS:
                timestamp = next(timestamps)
                message = next(messages)
            aligned_data[key].append(message)
    # change "dictionary of topics containing lists of messages" to "list of timestamps containing dictionary of messages" and return it
    return list(dict(zip(aligned_data.keys(), i)) for i in zip(*aligned_data.values()))


def load_anvil_annotation(anvil_file):
    # this is Rayan's code
    tree = etree.parse(anvil_file)
    root = tree.getroot()
    rows = []

    for anvil_track in root.iter("track"):
        trackname = anvil_track.attrib["name"]
        track_dict = {"Track": trackname}

        track_type = anvil_track.attrib["type"]
        if track_type == "primary":

            for anvil_el in anvil_track.iter("el"):
                timeframe_dict = {k.capitalize(): v for k, v in anvil_el.attrib.items()}
                feature_dict = {}
                for anvil_attribute in anvil_el.iter("attribute"):
                    feature_name = anvil_attribute.attrib["name"]
                    feature_value = anvil_attribute.text
                    feature_tempdict = {feature_name: feature_value}
                    feature_dict = {**feature_dict, **feature_tempdict}

                dict = {**track_dict, **timeframe_dict, **feature_dict}
                rows.append(dict)
        else:
            continue
    return rows


class ActionDataset(Dataset):
    def __init__(self, root_dir, ros_topic_filter=None, transform=None):
        self.transform = transform
        self.frames = []
        videos = []
        # find all directories containing valid data
        for root, dir, files in os.walk(root_dir):
            if os.path.basename(os.path.normpath(root)) == "frames":
                files.sort()
                base_directory = Path(root).parent
                videos.append({"base_directory": str(base_directory),
                               "annotation_file": os.path.join(base_directory, "annotation.anvil"),
                               "rosbag": os.path.join(base_directory, "recording.bag"),
                               "frames": list(
                                   {"path": os.path.join(root, file),
                                    "is_last": i == len(files) - 1,
                                    "sensor_data": {},
                                    "targets": {}} for i, file in enumerate(files))
                               })
        broken_videos = []
        annotation_map = OrderedDict()
        for i, video in enumerate(videos):
            total_frames = len(video["frames"])
            # load rosbag
            if not os.path.exists(video["rosbag"]):
                broken_videos.append(video["base_directory"])
                continue
            sensor_data = load_rosbag(video["rosbag"], ros_topic_filter)
            if not sensor_data or len(sensor_data[0]) != len(ros_topic_filter):
                broken_videos.append(video["base_directory"])
                continue
            sensor_data_length = len(sensor_data)
            if sensor_data_length == 0:
                broken_videos.append(video["base_directory"])
                continue
            if sensor_data_length < total_frames:
                video["frames"] = video["frames"][:sensor_data_length]
                video["frames"][-1]["is_last"] = True
                total_frames = sensor_data_length
            for j in range(total_frames):
                video["frames"][j]["sensor_data"] = torch.concat(list(sensor_data[j].values()))
            # load annotation
            if not os.path.exists(video["annotation_file"]):
                broken_videos.append(video["base_directory"])
                continue
            for annotation in load_anvil_annotation(video["annotation_file"]):
                track = str.lower(annotation["Track"])
                if track not in annotation_map:
                    annotation_map[track] = OrderedDict()
                for k in annotation:
                    if k not in ["Track", "Index", "Start", "End"]:
                        k_lc = str.lower(k)
                        if k_lc not in annotation_map[track]:
                            annotation_map[track][k_lc] = []
                        label = str.lower(annotation[k])
                        if label not in annotation_map[track][k_lc]:
                            annotation_map[track][k_lc].append(label)
                start = int(VIDEO_FPS * float(annotation["Start"]))
                end = int(VIDEO_FPS * float(annotation["End"]))
                if total_frames < end:
                    end = total_frames
                for j in range(start, end):
                    video["frames"][j]["targets"][str.lower(annotation["Track"])] = {
                        str.lower(k): str.lower(annotation[k]) for k in annotation if
                        k not in ["Track", "Index", "Start", "End"]}
            self.frames += video["frames"]
            print(f"\rLoading data ({int(((i + 1) / len(videos)) * 100)}%)", end="")
        log_message = f"\rLoaded {len(videos) - len(broken_videos)} videos. "
        if broken_videos:
            log_message += f"{broken_videos} were skipped due to missing sensor data and/or annotations."
        print(log_message)
        # create target encodings
        annotation_map = OrderedDict(sorted(annotation_map.items(), key=lambda t: t[0]))
        for track in annotation_map:
            annotation_map[track] = OrderedDict(sorted(annotation_map[track].items(), key=lambda t: t[0]))
            for label in annotation_map[track]:
                annotation_map[track][label].sort()
        for frame in self.frames:
            encoding = []
            for track in list(annotation_map.keys()):
                for label in annotation_map[track]:
                    actual_value = frame["targets"][track][label] if track in frame["targets"] and label in \
                                                                     frame["targets"][track] else ""
                    for possible_value in annotation_map[track][label]:
                        encoding.append(int(actual_value == possible_value))
            frame["targets"] = torch.tensor(encoding, device="cpu")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.value()

        sample = self.frames[index]
        sample["image"] = torchvision.io.read_image(sample["path"])

        if self.transform:
            sample = self.transform(sample)

        return sample
