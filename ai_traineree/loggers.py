import abc
from torch.utils.tensorboard import SummaryWriter
from typing import Dict

try:
    import neptune
except ImportError:
    pass


class DataLogger(abc.ABC):

    @abc.abstractmethod
    def log_value(self, name, value, step) -> None:
        pass

    @abc.abstractmethod
    def log_values_dict(self, name, values, step) -> None:
        pass

    @abc.abstractmethod
    def create_histogram(self, name, values, step) -> None:
        pass


class TensorboardLogger(DataLogger):
    """Tensorboard logger.

    Wrapper around the torch.utils.tensorboard.SummaryWriter.
    """

    def __init__(self, writer=None):
        if writer is None:
            writer = SummaryWriter()
        if not isinstance(writer, SummaryWriter):
            raise ValueError("Only `SummaryWriter` class is allowed for the Tensorboard logger")
        self.writer = writer

    def close(self):
        self.writer.close()

    def log_value(self, name: str, value, step: int) -> None:
        self.writer.add_scalar(name, value, step)

    def log_values_dict(self, name: str, values: Dict[str, float], step: int) -> None:
        self.writer.add_scalars(name, values, step)

    def create_histogram(self, name: str, values, step: int) -> None:
        self.writer.add_histogram(name, values, step)


class NeptuneLogger(DataLogger):
    """Neptune.ai logger.

    Wrapper around the Neptune.ai logger.
    """

    def __init__(self, project_name: str):
        self.project = neptune.init(project_name)
        pass

    def close(self):
        neptune.stop()

    def log_value(self, name: str, value, step: int) -> None:
        neptune.log_metric(name, x=step, y=value)

    def log_values_dict(self, name: str, values, step: int) -> None:
        for _name, _value in values.items():
            self.log_value(f"{name}/{_name}", _value, step)

    def create_histogram(self, name, values, step) -> None:
        return
