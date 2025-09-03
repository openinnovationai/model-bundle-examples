import os
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("NumPy not installed in this virtual environment. Skipping import.")

from model import Model


# Try to load the model.
le_model = Model(
    data_dir=os.path.realpath("."), config={"model_metadata": {"model_binary_dir": ""}}
)

le_model.load()

if le_model._model is None:
    raise RuntimeError("Model failed to load. Check the weights path")

# Test on different inputs based on the current example.
example_name = Path(".").resolve().name

if example_name == "fastai":
    test_input = np.ones((32, 32))
elif example_name == "pytorch":
    test_input = np.ones((1, 1, 28, 28))
elif example_name == "sklearn":
    test_input = [[1.0 for _ in range(10)]]
elif example_name == "statsmodels":
    test_input = [[1.0 for _ in range(11)]]
elif example_name == "tensorflow":
    test_input = np.random.rand(1, 28, 28)
elif example_name == "transformers":
    test_input = ["You are an excellent person!"]
elif example_name == "xgboost":
    test_input = [[1.0 for _ in range(10)]]

test_output = le_model.predict(test_input)

if test_output is None:
    raise RuntimeError("Prediction output is empty")
