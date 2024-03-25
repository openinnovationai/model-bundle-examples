from typing import Any
import os
import numpy as np
import joblib
import logging


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        config = kwargs["config"]
        model_metadata = config["model_metadata"]
        self._model_binary_dir = model_metadata["model_binary_dir"]
        self._model = None

    def load(self) -> None:
        model_binary_dir_path = os.path.join(
            str(self._data_dir), str(self._model_binary_dir)
        )
        pkl_filepath = os.path.join(model_binary_dir_path, "data", "model.pkl")
        logging.info(f"Loading model file {pkl_filepath}")
        self._model = joblib.load(pkl_filepath)

    def predict(self, model_input: Any) -> Any:
        x = np.asarray(model_input)  # Convert the REST input to a numpy array
        result = self._model.predict(x)
        predictions = result.tolist()  # Convert the model output to a Python list
        return {"predictions": predictions}
