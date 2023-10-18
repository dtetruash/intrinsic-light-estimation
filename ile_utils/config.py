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
    def get_config():
        if Config.__conf is None:
            Config.__conf = configparser.ConfigParser()
            Config.__conf.read(Config.__config_path)

            # Write in interpolated values.
            Config.__conf.set("paths", "scene_path", "%(dataset_path)s/%(scene)s")
            Config.__conf.set("paths", "data_path", "%(scene_path)s/%(split)s")

            if Config.__conf.has_option("paths", "split"):
                Config.__conf.set(
                    "paths",
                    "transforms_file",
                    "%(scene_path)s/transforms_%(split)s.json",
                )
            else:
                Config.__conf.set(
                    "paths", "transforms_file", "%(scene_path)s/transforms.json"
                )

            Config.__conf.set(
                "paths", "lights_file", "%(scene_path)s/lights_%(scene)s.json"
            )

        return Config.__conf
