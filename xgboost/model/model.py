from typing import Any, Optional
import os
import xgboost as xgb
import logging
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
        filepath = os.path.join(model_binary_dir_path, "data", "booster.json")
        logging.info(f"Loading model file {filepath}")
        self._model = xgb.Booster()
        self._model.load_model(filepath)

    def predict(self, model_input: Any) -> Any:
        assert self._model is not None
        input_xgb = xgb.DMatrix(np.asarray(model_input))
        res = self._model.predict(input_xgb)
        return {"predictions": res.tolist()}
