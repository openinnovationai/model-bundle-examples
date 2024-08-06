from typing import Any, Optional
import os
import numpy as np
import logging

import torch
from torch import nn
import torch.nn.functional as F


# CNN model class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(64 * 6 * 6, 10)

    def forward(self, x):
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = x.view(-1, 64 * 6 * 6)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        config = kwargs["config"]
        model_metadata = config["model_metadata"]
        self._model_binary_dir = model_metadata["model_binary_dir"]
        self._model: Optional[CNN] = None

    def load(self) -> None:
        model_binary_dir_path = os.path.join(
            str(self._data_dir), str(self._model_binary_dir)
        )
        filepath = os.path.join(model_binary_dir_path, "data", "model.pkl")
        logging.info(f"Loading model file {filepath}")
        self._model = CNN()
        self._model.load_state_dict(torch.load(filepath))
        self._model.eval()

    def predict(self, model_input: Any) -> Any:
        assert self._model is not None
        data = np.asarray(
            model_input, dtype=np.float32
        )  # Convert the REST input to a numpy array
        pred_tensor = self._model(torch.from_numpy(data))
        pred_np: np.ndarray = pred_tensor.detach().numpy()

        labels = pred_np.argmax(axis=1).tolist()
        return {"predictions": labels}
