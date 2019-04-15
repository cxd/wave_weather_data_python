import json
import io
import functools

class ReadConfig:

    def __init__(self):
        self.json_data = None

    def read_config(self, path):
        # Read configuration from file.
        with open(path, "r") as config_file:
            self.json_data = json.load(config_file)
        return self.json_data





