import logging
import os
import time
from typing import Any


class Model:
    def __init__(self, **kwargs) -> None:
        self._model_name = "custom_model"
        self._data_dir = kwargs.get("data_dir", None)
        self._config = kwargs.get("config", {})
        model_metadata = self._config.get("model_metadata", {})
        self._model_binary_dir = model_metadata.get("model_binary_dir", "")
        self._repeat_count = 1  # Default repeat count
        logging.info(f"Custom Model initialized with data_dir: {self._data_dir}")

    def load(self) -> None:
        logging.info("Custom model loaded (loads weights.bin for repeat count)")
        model_binary_dir_path = os.path.join(
            str(self._data_dir), str(self._model_binary_dir)
        )
        weights_filepath = os.path.join(model_binary_dir_path, "weights.bin")

        if not os.path.exists(weights_filepath):
            logging.warning(
                f"Weights file {weights_filepath} not found. Using default repeat_count=1"
            )

        logging.info(f"Loading model file {weights_filepath}")
        with open(weights_filepath, "r") as f:
            self._repeat_count = int(f.read().strip())

    def predict(self, model_input: Any) -> Any:
        # Simple echo prediction for demo; replace with your real logic
        text = model_input.get("text", "")
        # Repeat uppercased text
        prediction = "\n".join([text.upper()] * self._repeat_count)
        time.sleep(1)  # Simulate a delay
        return {"predictions": [prediction]}

    def custom_get(self) -> Any:
        return {"model_name": self._model_name, "repeat_count": self._repeat_count}

    def custom_post(self, model_input: Any) -> Any:
        text = model_input.get("text", "")
        text = "".join(
            c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(text)
        )
        # Repeat text according to weights
        text = "\n".join([text] * self._repeat_count)
        return {"predictions": text}
