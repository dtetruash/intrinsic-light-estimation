"""Read-in the configuration object lazily and give global access
to this object to other scopes
"""

import configparser
import io


class Config:
    __conf = None

    __config_path = "config.ini"

    @staticmethod
    def log_config(logger):
        with io.StringIO() as ss:
            logger.info("Configuration loaded from file was:")
            Config.get_config().write(ss)
            ss.seek(0)  # rewind
            logger.info(ss.read())

    @staticmethod
    def get_config(config_path=None):
        if Config.__conf is None:
            Config.__conf = configparser.ConfigParser()

            if config_path is None:
                config_path = Config.__config_path

            Config.__conf.read(config_path)

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
