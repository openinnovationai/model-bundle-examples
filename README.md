# OpenInnovation Module Bundle Examples

This repository contains examples to get started with the Open Innovation Model Bundle import.

## Installation

Clone the repository with `git clone https://github.com/openinnovationai/model-bundle-examples`

## Examples

Each folder has a Model Bundle created using a specific library.

To build a bundle for a specific example:

1. Move to the example folder
2. Optionally create a virtual environment: `python -m venv .venv`
3. Install the requirements: `pip install -r requirements.txt`
4. Run the `create_bundle.py` script
5. Upload on the Open Innovation Cluster Management (OICM) platform in the same directory

- the `model.py` model file,
- the `requirements.txt` file,
- the model weights (could be `model.pkl`, `model.pth`, `model.joblib`, etc.).

![Sample](docs/img/sample_bundle.png)

## Testing

Ensure you have Python 3.9 or higher and it is aliased to `python`. Below is what you should see

```
$ python --version
3.9.23
```

Test the code by running

```bash
bash tests/test_create_and_load.sh
```
