# Before you deploy

## Download a voice sample
This model bundle requires a voice sample. This example use the voice sample from https://github.com/vibevoice-community/VibeVoice/blob/4954a3f2d66372e3efd2a1726f9b420dc5c183c2/demo/voices/en-Alice_woman.wav.

Download this file and place it in the `data` folder before uploading this model bundle. Alternatively, record your own voice in `.wav` format.

## Disable SSL for Git
This model bundle requires some code not available on PyPI. Instead, it requires a package built from GitHub. Set the following environment variable in the deployment to disable SSL verification during requirements installation.

```
GIT_SSL_NO_VERIFY="true"
```
