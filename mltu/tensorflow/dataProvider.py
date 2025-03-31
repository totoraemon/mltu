import keras

from ..dataProvider import DataProvider as dataProvider

class DataProvider(dataProvider, keras.utils.Sequence):
    def __init__(self, *args, **kwargs):
        self.workers = kwargs.pop("workers", 10)
        self.use_multiprocessing = kwargs.pop("use_multiprocessing", False)
        self.max_queue_size = kwargs.pop("max_queue_size", 10)
        super().__init__(*args, **kwargs)
