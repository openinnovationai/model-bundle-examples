from typing import Any
import os
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import logging
import torch


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
        pkl_filepath = os.path.join(model_binary_dir_path, "data")
        logging.info(f"Loading model file {pkl_filepath}")
        self._model = AutoModelForSequenceClassification.from_pretrained(
            "finiteautomata/bertweet-base-sentiment-analysis"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(pkl_filepath)

    def predict(self, model_input: Any) -> Any:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        if isinstance(model_input, list):
            model_input = model_input[0]

        inputs = self._tokenizer(
            model_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        outputs = self._model(input_ids, attention_mask=attention_mask)
        predicted_label = torch.argmax(outputs.logits).item()
        sentiment = "positive" if predicted_label == 1 else "negative"
        return {"predictions": sentiment}
