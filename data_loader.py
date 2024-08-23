from copy import copy

import numpy as np
import torch
import os
import xml.etree.ElementTree as ET

import torchvision.io
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from collections import OrderedDict
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


def load_annotation_map(specfile):
    xml_root = ET.parse(specfile).getroot()
    assert xml_root.tag == "annotation-spec"
    # get value sets
    value_sets = {}
    for value_set in xml_root.iter("valueset"):
        name = str.lower(value_set.attrib["name"])
        value_sets[name] = []
        for value in value_set.iter("value-el"):
            value_sets[name].append(str.lower(value.text))
        value_sets[name].sort()
    # get tracks
    annotation_map = OrderedDict()

    def parse_track(track_name, track):
        track_name += str.lower(track.attrib["name"])
        annotation_map[track_name] = OrderedDict()
        for attribute in track.iter("attribute"):
            attribute_name = str.lower(attribute.attrib["name"])
            if "valuetype" in attribute.attrib:
                annotation_map[track_name][attribute_name] = value_sets[str.lower(attribute.attrib["valuetype"])].copy()
            else:
                annotation_map[track_name][attribute_name] = []
                for value in attribute.iter("value-el"):
                    annotation_map[track_name][attribute_name].append(str.lower(value.text))
            annotation_map[track_name][attribute_name].append("not present")

    for element in xml_root.find("body"):
        if element.tag == "group":
            track_name = str.lower(element.attrib["name"]) + "."
            for track in element.iter("track-spec"):
                parse_track(copy(track_name), track)
        elif element.tag == "track-spec":
            parse_track("", element)
    return annotation_map


class Video(Dataset):
    def __init__(self, path, annotation_map, ros_topics=None, transform=None):
        self.transform = transform
        self.frames = []

        annotation_file = os.path.join(path, "annotation.anvil")
        frames_dir = os.path.join(path, "frames")
        if not os.path.exists(annotation_file) or not os.path.exists(frames_dir):
            raise FileNotFoundError()
        for file in os.listdir(frames_dir):
            file_full_path = os.path.join(frames_dir, file)
            if os.path.isfile(file_full_path) and file.endswith(".png"):
                self.frames.append({"path": os.path.join(frames_dir, file),
                                    "sensor_data": {},
                                    "targets": {}})
        if ros_topics is not None:
            # load rosbag
            ros_bag = os.path.join(path, "recording.bag")
            if not os.path.exists(ros_bag):
                raise FileNotFoundError()
            sensor_data = load_rosbag(ros_bag, ros_topics)
            if not sensor_data or len(sensor_data[0]) != len(ros_topics):
                raise ValueError()
            sensor_data_length = len(sensor_data)
            if sensor_data_length == 0:
                raise ValueError()
            if sensor_data_length < len(self.frames):
                self.frames = self.frames[:sensor_data_length]
            for j in range(len(self.frames)):
                self.frames[j]["sensor_data"] = torch.concat(list(sensor_data[j].values())).to(torch.float32)
        # load annotation
        if not os.path.exists(annotation_file):
            raise FileNotFoundError()
        for annotation in load_anvil_annotation(annotation_file):
            start = int(VIDEO_FPS * float(annotation["Start"]))
            end = int(VIDEO_FPS * float(annotation["End"]))
            if len(self.frames) < end:
                end = len(self.frames)
            for j in range(start, end):
                self.frames[j]["targets"][str.lower(annotation["Track"])] = {
                    str.lower(k): str.lower(annotation[k]) for k in annotation if
                    k not in ["Track", "Index", "Start", "End"]}

        # create target encodings
        for frame in self.frames:
            encoding = []
            for track in annotation_map.keys():
                for label in annotation_map[track]:
                    actual_value = frame["targets"][track][label] if track in frame["targets"] and label in \
                                                                     frame["targets"][track] else ""
                    not_present = True
                    for possible_value in annotation_map[track][label]:
                        if possible_value == "not present":
                            encoding.append(int(not_present))
                        else:
                            equals = actual_value == possible_value
                            not_present = not_present and not equals
                            encoding.append(int(equals))
            frame["targets"] = torch.tensor(encoding, device="cpu", dtype=torch.float32)
        self.target_size = self.frames[0]["targets"].size(0)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.value()
        sample = self.frames[index]
        sample["image"] = torchvision.io.read_image(sample["path"]).to(torch.float32)
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        return sample


def load_data(root_dir, annotation_map, ros_topics=None, batch_size=1, transform=None, shuffle=False, concat=False):
    # find all directories containing valid data
    videos_dirs = []
    loaded_videos = []
    broken_videos = []
    for root, dir, files in os.walk(root_dir):
        if os.path.basename(os.path.normpath(root)) == "frames":
            files.sort()
            base_directory = Path(root).parent
            videos_dirs.append(base_directory)
    # load videos
    for dir in videos_dirs:
        try:
            loaded_videos.append(Video(dir, annotation_map, ros_topics=ros_topics, transform=transform))
            print(f"\rLoading data: {int((len(loaded_videos) + len(broken_videos)) / len(videos_dirs) * 100)}%", end="")
        except (FileNotFoundError, ValueError):
            broken_videos.append(dir)
    log_string = f"\rLoaded data in {root_dir}"
    if len(broken_videos):
        log_string += f" :{len(broken_videos)} of {len(broken_videos) + len(loaded_videos)} failed."
    print(log_string)
    if concat:
        loaded_videos = [ConcatDataset(loaded_videos)]
    return (DataLoader(video, batch_size=batch_size, shuffle=shuffle) for video in loaded_videos)
