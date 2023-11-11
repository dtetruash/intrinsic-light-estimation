import json

import data_loaders.olat_render as ro
from ile_utils.config import Config

# TODO: move all metadata methods here


def get_transform(frame_num, split="val"):
    config = Config.get_config()
    file_path = config.get("dataset", "scene_path") + f"/transforms_{split}.json"
    with open(file_path, "r") as tf:
        frame_transforms = json.loads(tf.read())

    return ro.get_c2w(frame_num, frame_transforms)


def get_camera_orientation(frame_num, split="val"):
    return ro.to_rotation(get_transform(frame_num, split))
