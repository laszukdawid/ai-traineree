import json
import time
from os.path import getsize, splitext

from .data_logger import DataLogger


class FileLogger(DataLogger):
    """File Logger

    Write logs into a file in csv-like format.
    Each write is a separate line initiated with timestamp (either provided, or current).
    All rows have "{time},{key1},{value1},{key2},{value2},..." .

    Parameters:
        filepath (str): Exact filepath where the logs are stored. Hyperparameters are stored with the same path
            but with different extension, i.e. `_hparams.json`.
        max_file_size_kb (int): Maximum size (in kB) of the log file. On exceeding the limit,
            the earliest half is removed. Defaults to 1 MB.

    """

    name = "FileLogger"

    def __init__(self, filepath: str, max_file_size_kb: int = 1024):
        """

        Parameters:
            filepath: Where to store the file.
            max_filesize_kb: Maximum size of the file to keep locally in kilobytes. Default: 1024 kB.

        """

        self.filepath = filepath
        self.max_size = max_file_size_kb * 1024
        self.__init_file()

    def __str__(self):
        return self.name

    def __init_file(self):
        open(self.filepath, "a").close()

    def close(self) -> None:
        return super().close()

    @staticmethod
    def _timestamp():
        return time.time()

    def _check_and_trim(self):
        current_size = getsize(self.filepath)
        if current_size < self.max_size:
            return

        # File size too big - trim the first half
        with open(self.filepath, "w+") as f:
            all_rows = f.readlines()
            f.writelines(all_rows[len(all_rows) // 2 :])

    def set_hparams(self, *, hparams: dict):
        filepath = splitext(self.filepath)[0]

        with open(filepath + "_hparams.json", "w") as f:
            json.dump(hparams, f)

    def log_value(self, name: str, value, step: int) -> None:
        self._check_and_trim()
        with open(self.filepath, "a") as f:
            f.write(f"{self._timestamp()},step,{step},{name},{value}\n")

    def log_values_dict(self, name: str, values: dict[str, float], step: int) -> None:
        self._check_and_trim()
        log = ",".join([f"{name}_{key},{value}" for (key, value) in values.items()])
        with open(self.filepath, "a") as f:
            f.write(f"{self._timestamp()},{log},step,{step}\n")

    def add_histogram(self, *args, **kwargs):
        """Logs provided histogram with all its parameters. Note that the input is alread a histogram."""
        ...

    def create_histogram(self, name: str, values, step: int) -> None:
        """Creates a histogram out of provided data."""
        ...
