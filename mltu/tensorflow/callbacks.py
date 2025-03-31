import os
import typing
import tensorflow as tf
import keras

from pathlib import Path
import logging

class Model2onnx(keras.callbacks.Callback):
    """ Converts the model to onnx format after training is finished. """
    def __init__(
        self, 
        saved_model_path: str, 
        metadata: typing.Union[dict, None]=None,
        save_on_epoch_end: bool=False,
        opset: typing.Union[int, None]=None
        ) -> None:
        """ Converts the model to onnx format after training is finished.
        Args:
            saved_model_path (str): Path to the saved model.
            metadata (dict, optional): Dictionary containing metadata to be added to the onnx model. Defaults to None.
            save_on_epoch_end (bool, optional): Save the onnx model on every epoch end. Defaults to False.
        """
        super().__init__()
        self.saved_model_path = saved_model_path
        self.metadata = metadata
        self.save_on_epoch_end = save_on_epoch_end
        self.opset = opset

        try:
            import tf2onnx
        except:
            raise ImportError("tf2onnx is not installed. Please install it using 'pip install tf2onnx'")
        
        try:
            import onnx
        except:
            raise ImportError("onnx is not installed. Please install it using 'pip install onnx'")

    @staticmethod
    def model2onnx(model: keras.Model, onnx_model_path: str, opset: typing.Union[int, None]=None):
        import tf2onnx

        # Handle input signature where model has multiple inputs
        input_signature = [tf.TensorSpec(shape=t.shape, dtype=t.dtype, name=t.name) for t in model.inputs] # pyright: ignore

        # convert the model to onnx format
        model_proto, external_tensor_storage = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=opset, output_path=onnx_model_path)

        return model_proto, external_tensor_storage

    @staticmethod
    def include_metadata(onnx_model_path: str, metadata: typing.Union[dict, None]=None):
        if metadata and isinstance(metadata, dict):

            import onnx
            # Load the ONNX model
            onnx_model = onnx.load(onnx_model_path)

            # Add the metadata dictionary to the model's metadata_props attribute
            for key, value in metadata.items():
                meta = onnx_model.metadata_props.add()
                meta.key = key
                meta.value = str(value)

            # Save the modified ONNX model
            onnx.save(onnx_model, onnx_model_path)

    def on_epoch_end(self, epoch: int, logs: typing.Union[dict, None]=None):
        """ Converts the model to onnx format on every epoch end. """
        if self.save_on_epoch_end:
            self.on_train_end(logs=logs)

    def on_train_end(self, logs: typing.Union[dict, None]=None):
        """ Converts the model to onnx format after training is finished. """
        self._model.load_weights(self.saved_model_path) # pyright: ignore
        onnx_model_path = str(Path(self.saved_model_path).with_suffix('.onnx'))
        self.model2onnx(self.model, onnx_model_path, self.opset) # pyright: ignore
        self.include_metadata(onnx_model_path, self.metadata)


class TrainLogger(keras.callbacks.Callback):
    """Logs training metrics to a file.
    
    Args:
        log_path (str): Path to the directory where the log file will be saved.
        log_file (str, optional): Name of the log file. Defaults to 'logs.log'.
        logLevel (int, optional): Logging level. Defaults to logging.INFO.
    """
    def __init__(self, log_path: str, log_file: str="logs.log", logLevel=logging.INFO, console_output=False) -> None:
        super().__init__()
        self.log_path = log_path
        self.log_file = log_file

        if not os.path.exists(log_path):
            os.mkdir(log_path)

        self.logger = logging.getLogger()
        self.logger.setLevel(logLevel)

        self.formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        self.file_handler = logging.FileHandler(os.path.join(self.log_path, self.log_file))
        self.file_handler.setLevel(logLevel)
        self.file_handler.setFormatter(self.formatter)

        if not console_output:
            self.logger.handlers[:] = []

        self.logger.addHandler(self.file_handler)

    def on_epoch_end(self, epoch: int, logs: typing.Union[dict, None]=None):
        if logs:
            epoch_message = f"Epoch {epoch}; "
            logs_message = "; ".join([f"{key}: {value}" for key, value in logs.items()])
            self.logger.info(epoch_message + logs_message)


class WarmupCosineDecay(keras.callbacks.Callback):
    """ Cosine decay learning rate scheduler with warmup

    Args:
        lr_after_warmup (float): Learning rate after warmup
        final_lr (float): Final learning rate
        warmup_epochs (int): Number of warmup epochs
        decay_epochs (int): Number of decay epochs
        initial_lr (float, optional): Initial learning rate. Defaults to 0.0.
        verbose (bool, optional): Whether to print learning rate. Defaults to False.
    """
    def __init__(
            self, 
            lr_after_warmup: float, 
            final_lr: float, 
            warmup_epochs: int, 
            decay_epochs: int, 
            initial_lr: float=0.0, 
            verbose=False
        ) -> None:
        super(WarmupCosineDecay, self).__init__()
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.initial_lr = initial_lr
        self.verbose = verbose
        self.model: keras.Model

    def on_epoch_begin(self, epoch: int, logs: typing.Union[dict, None]=None):
        """ Adjust learning rate at the beginning of each epoch """

        if epoch >= self.warmup_epochs + self.decay_epochs:
            return

        if epoch < self.warmup_epochs:
            lr = self.initial_lr + (self.lr_after_warmup - self.initial_lr) * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / self.decay_epochs
            lr = self.final_lr + 0.5 * (self.lr_after_warmup - self.final_lr) * (1 + tf.cos(tf.constant(progress) * 3.14159)) # pyright: ignore

        tf.keras.backend.set_value(self.model.optimizer.lr, lr) # pyright: ignore
        
        if self.verbose:
            print(f"Epoch {epoch + 1} - Learning Rate: {lr}")
    
    def on_epoch_end(self, epoch: int, logs: typing.Union[dict, None]=None):
        logs = logs or {}
        
        # Log the learning rate value
        logs["lr"] = self.model.optimizer.lr