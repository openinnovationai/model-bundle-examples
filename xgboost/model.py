import logging
import os
from typing import Any, Optional

import xgboost as xgb
import numpy as np


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        config = kwargs["config"]
        model_metadata = config["model_metadata"]
        self._model_binary_dir = model_metadata["model_binary_dir"]
        self._model: Optional[xgb.Booster] = None

    def load(self) -> None:
        model_binary_dir_path = os.path.join(
            str(self._data_dir), str(self._model_binary_dir)
        )
        weights_filepath = os.path.join(model_binary_dir_path, "booster.json")
        logging.info(f"Loading model file {weights_filepath}")
        self._model = xgb.Booster()
        self._model.load_model(weights_filepath)

    def predict(self, model_input: Any) -> Any:
        if self._model is None:
            raise ValueError("Model not loaded")

        input_xgb = xgb.DMatrix(np.asarray(model_input))
        res = self._model.predict(input_xgb)
        return {"predictions": res.tolist()}
