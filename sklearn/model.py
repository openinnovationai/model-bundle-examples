import logging
import os
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        config = kwargs["config"]
        model_metadata = config["model_metadata"]
        self._model_binary_dir = model_metadata["model_binary_dir"]
        self._model: Optional[RandomForestRegressor] = None

    def load(self) -> None:
        model_binary_dir_path = os.path.join(
            str(self._data_dir), str(self._model_binary_dir)
        )
        joblib_filepath = os.path.join(model_binary_dir_path, "data", "model.joblib")
        logging.info(f"Loading model file {joblib_filepath}")
        self._model = joblib.load(joblib_filepath)

    def predict(self, model_input: Any) -> Any:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        x = np.asarray(model_input)  # Convert the REST input to a numpy array
        result = self._model.predict(x)
        predictions = result.tolist()  # Convert the model output to a Python list
        return {"predictions": predictions}
