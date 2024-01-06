import time


class Metric(object):
    def __init__(self) -> None:
        self.start_ts = 0
        self.metrics = []

    def start(self):
        self.start_ts = time.time()

    def emit(self, module, output):
        d = time.time() - self.start_ts
        self.metrics.append((module._get_name(), d, output.reshape((-1, 1)).size()[0]))
        return


metric = Metric()
