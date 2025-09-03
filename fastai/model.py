import logging
import os
from typing import Any, Optional

import numpy as np
from fastai.learner import load_learner, Learner


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        config = kwargs["config"]
        model_metadata = config["model_metadata"]
        self._model_binary_dir = model_metadata["model_binary_dir"]
        self._model: Optional[Learner] = None

    def load(self) -> None:
        model_binary_dir_path = os.path.join(
            str(self._data_dir), str(self._model_binary_dir)
        )
        filepath = os.path.join(model_binary_dir_path, "model.fastai")
        logging.info(f"Loading model file {filepath}")
        self._model = load_learner(filepath)

    def predict(self, model_input: Any) -> Any:
        if self._model is None:
            raise ValueError("Model not loaded")
        # Convert the REST input to a numpy array
        x = np.asarray(model_input, dtype=np.uint8)
        prediction = self._model.predict(x)
        label = prediction[0]
        return {"predictions": label}
