from typing import Any, Optional
import os
import tensorflow as tf
import logging
import numpy as np


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        config = kwargs["config"]
        model_metadata = config["model_metadata"]
        self._model_binary_dir = model_metadata["model_binary_dir"]
        self._model: Optional[tf.keras.models.Sequential] = None

    def _load_safe(self) -> None:
        model_binary_dir_path = os.path.join(
            str(self._data_dir), str(self._model_binary_dir)
        )
        pkl_filepath = os.path.join(
            model_binary_dir_path, "data", "model_tf.weights.h5"
        )
        logging.info(f"Loading model file {pkl_filepath}")
        self._model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10),
            ]
        )
        self._model.load_weights(pkl_filepath)

    def load(self) -> None:
        try:
            self._load_safe()
        except Exception as e:
            print(e)
            raise e

    def predict(self, model_input: Any) -> Any:
        assert self._model is not None
        np_input = np.asarray(model_input, dtype=np.uint8)
        result: np.ndarray = self._model.predict(np_input)
        prediction = result.argmax(axis=1)
        return {"predictions": prediction}
