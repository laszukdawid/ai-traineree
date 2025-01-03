import logging

from .data_logger import DataLogger

try:
    import neptune
except ImportError:
    msg = (
        "Couldn't import `neptune` module. This likely means that "
        "the `neptune` isn't installed, or that you're using a different Python environment. "
        "Installing `neptune` can be done for example by executing `pip install neptune`."
    )
    logging.exception(msg)
    raise ImportError(msg)


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

    def set_hparams(self, *, hparams: dict): ...

    def log_value(self, name: str, value, step: int) -> None:
        self.experiment.log_metric(name, x=step, y=value)

    def log_values_dict(self, name: str, values, step: int) -> None:
        for _name, _value in values.items():
            self.log_value(f"{name}/{_name}", _value, step)

    def add_histogram(self, *args, **kwargs) -> None:
        return

    def create_histogram(self, name, values, step) -> None:
        return
