import abc


class DataLogger(abc.ABC):
    def __dell__(self):
        self.close()

    @abc.abstractmethod
    def close(self) -> None: ...

    @abc.abstractmethod
    def set_hparams(self, *, hparams: dict) -> None: ...

    @abc.abstractmethod
    def log_value(self, name, value, step) -> None: ...

    @abc.abstractmethod
    def log_values_dict(self, name, values, step) -> None: ...

    @abc.abstractmethod
    def add_histogram(self, *args, **kwargs) -> None: ...

    @abc.abstractmethod
    def create_histogram(self, name, values, step) -> None:
        """Creates a histogram out of provided data."""
        ...
