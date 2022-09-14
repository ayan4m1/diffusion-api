# diffusion-api

This package provides an HTTP API for generating Stable Diffusion images programmatically.

## environment

 * Python 3
 * [Stable Diffusion v1.4 Model]()

Install dependencies using:

> pip install -r requirements.txt
>
> pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

Run it using:

> flask --app main.py run