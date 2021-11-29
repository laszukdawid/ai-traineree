import abc
import logging
from typing import Dict

import_warn = []
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    import_warn.append("Tensorboard")

try:
    import neptune
except ImportError:
    import_warn.append("Neptune")

if import_warn:
    pkgs = " and ".join(import_warn)
    logging.warning("%s not installed", pkgs)


class DataLogger(abc.ABC):
    def __dell__(self):
        self.close()

    @abc.abstractmethod
    def close(self) -> None:
        pass

    @abc.abstractmethod
    def set_hparams(self, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def log_value(self, name, value, step) -> None:
        pass

    @abc.abstractmethod
    def log_values_dict(self, name, values, step) -> None:
        pass

    @abc.abstractmethod
    def add_histogram(self, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def create_histogram(self, name, values, step) -> None:
        """Creates a histogram out of provided data."""
        pass


class TensorboardLogger(DataLogger):
    """Tensorboard logger.

    Wrapper around the torch.utils.tensorboard.SummaryWriter.
    """

    name = "TensorboardLogger"

    def __init__(self, writer=None, *args, **kwargs):
        """
        If no SummaryWriter writer is proved then one is intiatied with provided parameters.

        Parameters:
            writer: (Optional) SummaryWriter instance.
                If not provided then one is created using `torch.utils.tensorboard.SummaryWriter`.

        """
        if writer is None:
            writer = SummaryWriter(*args, **kwargs)
        if not isinstance(writer, SummaryWriter):
            raise ValueError("Only `SummaryWriter` class is allowed for the Tensorboard logger")
        self.writer = writer

    def __str__(self):
        return self.name

    def close(self):
        self.writer.close()

    def set_hparams(self, *args, **kwargs):
        self.writer.add_hparams(*args, **kwargs)

    def log_value(self, name: str, value, step: int) -> None:
        self.writer.add_scalar(name, value, step)

    def log_values_dict(self, name: str, values: Dict[str, float], step: int) -> None:
        self.writer.add_scalars(name, values, step)

    def add_histogram(self, *args, **kwargs):
        """Logs provided histogram with all its parameters. Note that the input is alread a histogram."""
        self.writer.add_histogram_raw(*args, **kwargs)

    def create_histogram(self, name: str, values, step: int) -> None:
        """Creates a histogram out of provided data."""
        self.writer.add_histogram(name, values, step)


class NeptuneLogger(DataLogger):
    """Neptune.ai logger.

    Wrapper around the Neptune.ai logger.
    """

    name = "NeptuneLogger"

    def __init__(self, project_name: str, **kwargs):
        params = kwargs.pop("params", None)
        self.project = neptune.init(project_name, **kwargs)
        self.experiment = neptune.create_experiment(params=params)

    def __str__(self) -> str:
        return self.name

    def close(self):
        self.experiment.stop()

    def set_hparams(self, *args, **kwargs):
        pass

    def log_value(self, name: str, value, step: int) -> None:
        self.experiment.log_metric(name, x=step, y=value)

    def log_values_dict(self, name: str, values, step: int) -> None:
        for _name, _value in values.items():
            self.log_value(f"{name}/{_name}", _value, step)

    def add_histogram(self, *args, **kwargs) -> None:
        return

    def create_histogram(self, name, values, step) -> None:
        return
