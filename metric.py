import time

class Metric(object):
    """
    Metric class to record and store performance metrics.

    Attributes:
        start_ts (float): The timestamp when the metric recording started.
        metrics (list): A list to store recorded metrics as tuples (module_name, duration, output_size).

    Methods:
        __init__: Initializes a Metric object with an empty metrics list.
        start: Records the current timestamp as the start time.
        emit: Records a metric entry with module name, duration, and output size.

    Usage:
        metric = Metric()
        metric.start()
        # Perform some operation
        output = ...
        metric.emit(module, output)
        # Record more metrics if needed
    """

    def __init__(self) -> None:
        """
        Initializes a Metric object with an empty metrics list.
        """
        self.start_ts = 0
        self.metrics = []

    def start(self):
        """
        Records the current timestamp as the start time.
        """
        self.start_ts = time.time()

    def emit(self, module, output):
        """
        Records a metric entry with module name, duration, and output size.

        Args:
            module: The module or component being measured.
            output: The output of the module's operation.

        Returns:
            None
        """
        # Calculate the duration of the operation
        d = time.time() - self.start_ts
        # Record the metric as a tuple (module_name, duration, output_size)
        self.metrics.append((module._get_name(), d, output.reshape((-1, 1)).size()[0]))
