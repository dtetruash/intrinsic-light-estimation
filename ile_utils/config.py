"""Read-in the configuration object lazily and give global access
to this object to other scopes
"""

import configparser
import io


class Config:
    __conf = None

    __config_path = "config.ini"
    __loaded_path = None

    @staticmethod
    def log_config(logger=None):
        if logger is None:
            out_func = print
        else:
            out_func = logger.info

        with io.StringIO() as ss:
            Config.get_config().write(ss)
            ss.seek(0)  # rewind

            out_func(f"Configuration loaded from file {Config.__loaded_path} was:")
            out_func(ss.read())

    @staticmethod
    def get_config(config_path=None):
        if Config.__conf is None:
            Config.__conf = configparser.ConfigParser()

            if config_path is None:
                config_path = Config.__config_path

            Config.__conf.read(config_path)
            Config.__loaded_path = config_path

            # Write in interpolated values.
            Config.__conf.set("dataset", "scene_path", "%(dataset_path)s/%(scene)s")
            Config.__conf.set("dataset", "data_path", "%(scene_path)s/%(split)s")

            if Config.__conf.has_option("dataset", "split"):
                Config.__conf.set(
                    "dataset",
                    "transforms_file",
                    "%(scene_path)s/transforms_%(split)s.json",
                )
            else:
                Config.__conf.set(
                    "dataset", "transforms_file", "%(scene_path)s/transforms.json"
                )

            Config.__conf.set(
                "dataset", "lights_file", "%(scene_path)s/lights_%(scene)s.json"
            )

        return Config.__conf
