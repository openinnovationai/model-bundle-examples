# Custom endpoints example

This folder contains a simple example showing how to add custom endpoints to your model bundle.

## Defining custom endpoints

Endpoints are defined in `endpoints.yaml`. Each endpoint specifies:
- The path (endpoint)
- The HTTP method (GET or POST)
- The Python method to call (function_name)

Example from `endpoints.yaml`:

```
endpoints:
  - endpoint: /custom-get
    http_method: GET
    function_name: custom_get
  - endpoint: /custom-post
    http_method: POST
    function_name: custom_post
```

## Loading weights

This example loads a simple weight from `weights.bin`. The value in this file controls how many times a string is repeated in predictions. For example, if the file contains the number `3`, the predicted string is repeated three times.

This file is analogous to actual model weights that can be loaded into a Scikit Learn or PyTorch model, for instance.

## Example functions

Custom endpoint logic lives in `model.py`. This example provides two functions:

- `custom_get()`: Returns the model name and repeat count.
- `custom_post(model_input)`: Returns a text transformation, repeated as many times as the value in `weights.bin`.

You can adjust these functions to add your own logic.

## Usage

1. Set up your backend to load this model bundle.
2. Make a GET request to `/custom-get` or a POST request to `/custom-post` with a JSON payload.
3. The endpoints will use the code in `model.py` and the value in `weights.bin` to respond.

## Files

- `endpoints.yaml`: Defines the custom endpoints.
- `model.py`: Contains logic for the endpoints.
- `weights.bin`: Controls repeat count (change this number and reload for different behavior).
