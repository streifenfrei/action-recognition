import argparse

from data_loader import ActionDataset

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("path")
    argparser.add_argument("-r", "--ros", action="append")
    args = argparser.parse_args()
    # e.g. python ./main.py /path/to/dataset -r /eit_data_PCB4 -r /eit_data_PCB5
    dataset = ActionDataset(args.path,
                            ros_topic_filter=args.ros)
    # TODO use the data
