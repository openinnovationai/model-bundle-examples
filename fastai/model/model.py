from typing import Any, Optional
import os
import numpy as np
import logging
from fastai.learner import load_learner, Learner
from PIL import Image


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
        filepath = os.path.join(model_binary_dir_path, "data", "model.fastai")
        logging.info(f"Loading model file {filepath}")
        self._model = load_learner(filepath)

    def predict(self, model_input: Any) -> Any:
        assert self._model is not None
        x = np.asarray(
            model_input, dtype=np.uint8
        )  # Convert the REST input to a numpy array
        label, _, _ = self._model.predict(x)
        return {"predictions": label}
