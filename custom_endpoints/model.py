import logging
import time
from typing import Any


class Model:
    def __init__(self, **kwargs) -> None:
        self._model_name = "custom_model"
        self._data_dir = kwargs.get("data_dir", None)
        self._config = kwargs.get("config", {})
        logging.info(f"Custom Model initialized with data_dir: {self._data_dir}")

    def load(self) -> None:
        logging.info("Custom model loaded (noop in demo)")

    def predict(self, model_input: Any) -> Any:
        # Simple echo prediction for demo; replace with your real logic
        text = model_input.get("text", "")
        prediction = text.upper()  # Just uppercasing as a sample
        time.sleep(1)  # Simulate a delay
        return {"predictions": [prediction]}

    def custom_get(self) -> Any:
        return {"model_name": self._model_name}

    def custom_post(self, model_input: Any) -> Any:
        text = model_input.get("text", "")
        text = "".join(
            c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(text)
        )
        return {"predictions": text}
