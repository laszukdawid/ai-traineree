import logging

from .data_logger import DataLogger

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    msg = (
        "Couldn't import `SummaryWriter` from `torch.utils.tensorboard`. This likely means that "
        "the Tensorboard isn't installed, or that you're using a different Python environment. "
        "Installing Tensorboard can be done for example by executing `pip install tensorboard`."
    )
    logging.exception(msg)
    raise ImportError(msg)


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

    def set_hparams(self, *, hparams: dict):
        self.writer.add_hparams(hparam_dict=hparams, metric_dict={})

    def log_value(self, name: str, value, step: int) -> None:
        self.writer.add_scalar(name, value, step)

    def log_values_dict(self, name: str, values: dict[str, float], step: int) -> None:
        self.writer.add_scalars(name, values, step)

    def add_histogram(self, *args, **kwargs):
        """Logs provided histogram with all its parameters. Note that the input is alread a histogram."""
        self.writer.add_histogram_raw(*args, **kwargs)

    def create_histogram(self, name: str, values, step: int) -> None:
        """Creates a histogram out of provided data."""
        self.writer.add_histogram(name, values, step)
