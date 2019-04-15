import os as os
from pandas import pandas as p
from pandas.compat import StringIO, BytesIO


class ReadCsv:

    def __init__(self, base_dir):
        self.base_dir = base_dir

    @staticmethod
    def all_files(base_path):
        all_files = []
        for file in os.scandir(base_path):
            if file.is_file():
                all_files.append(file.name)
        return all_files

    def read_csv(self, file_path, skip_lines=0, colnames=[]):
        if colnames.__len__() > 0:
            temp = p.read_csv(file_path, skiprows=skip_lines, header=None, names=colnames)
        else:
            temp = p.read_csv(file_path, skiprows=skip_lines)
        return temp

    def process_directory(self, path="", skip_lines=0, colnames=[]):
        base_path = self.base_dir
        if path != "":
            base_path = os.path.join(base_path, path)
        files = ReadCsv.all_files(base_path)
        data = None
        all_paths = map(lambda file: os.path.join(base_path, file), files)
        for file_path in all_paths:
            temp = self.read_csv(file_path, skip_lines, colnames)
            if data is None:
                data = temp
            else:
                data = p.concat([data, temp])
        return data
