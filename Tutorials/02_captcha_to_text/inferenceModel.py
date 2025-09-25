import os
import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from model import train_model
import tensorflow as tf
from mltu.utils.text_utils import ctc_decoder, get_cer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    configs = BaseModelConfigs.load("Models/02_captcha_to_text/202509242132/configs.yaml")

    # Try ONNX inference first; if the .onnx does not exist or loading fails,
    # fall back to using the Keras model directly from the saved .h5 weights.
    onnx_file = f"{configs.model_path}/model.onnx"
    keras_h5 = f"{configs.model_path}/model.h5"
    model = None
    use_onnx = False
    try:
        if os.path.exists(onnx_file):
            model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
            use_onnx = True
        else:
            raise FileNotFoundError('ONNX model not found')
    except Exception:
        # Fall back to Keras
        print('Falling back to Keras model inference using', keras_h5)
        # Reconstruct architecture and load weights
        vocab = configs.vocab
        height = configs.height if hasattr(configs, 'height') else 50
        width = configs.width if hasattr(configs, 'width') else 200
        input_dim = (height, width, 3)
        keras_model = train_model(input_dim=input_dim, output_dim=len(vocab))
        keras_model.load_weights(keras_h5)

        class KerasImageToWord:
            def __init__(self, keras_model, char_list, input_shape):
                self.model = keras_model
                self.char_list = char_list
                self.input_shape = input_shape

            def predict(self, image: np.ndarray):
                image = cv2.resize(image, self.input_shape[0:2][::-1])
                image_pred = np.expand_dims(image, axis=0).astype(np.float32) / 255.0
                preds = self.model.predict(image_pred)
                text = ctc_decoder(preds, self.char_list)[0]
                return text

        model = KerasImageToWord(keras_model, configs.vocab, input_dim)

    df = pd.read_csv("Models/02_captcha_to_text/202509242132/val.csv").values.tolist()

    accum_cer = []
    for image_path, label in tqdm(df):
        image = cv2.imread(image_path.replace("\\", "/"))

        prediction_text = model.predict(image)

        cer = get_cer(prediction_text, label)
        print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")

        accum_cer.append(cer)

    print(f"Average CER: {np.average(accum_cer)}")