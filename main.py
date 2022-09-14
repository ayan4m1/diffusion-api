import torch
import sys
import cv2
import RRDBNet_arch as arch
import numpy as np

from os import environ
from torch import autocast
from dotenv import load_dotenv
from piexif import dump, ExifIFD
from PIL import Image, PngImagePlugin
from flask import Flask, request, send_file
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, LMSDiscreteScheduler

# load settings from .env file
load_dotenv('.env')

sd_model_path = environ.get('SD_MODEL_PATH')

if not sd_model_path:
    print('No SD_MODEL_PATH found in .env file!')
    exit(1)

esrgan_model_path = environ.get('ESRGAN_MODEL_PATH')


# create shared objects
app = Flask(__name__)
device = torch.device('cuda')

# create Stable Diffusion pipelines
lms = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear"
)
text_pipe = StableDiffusionPipeline.from_pretrained(sd_model_path, scheduler=lms, revision="fp16", torch_dtype=torch.float16)
text_pipe = text_pipe.to(device)
text_pipe.enable_attention_slicing()
# image_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(sd_model_path, scheduler=lms, revision="fp16", torch_dtype=torch.float16)
# image_pipe = image_pipe.to(device)
# image_pipe.enable_attention_slicing()

# create ESRGAN model
upscale_model = arch.RRDBNet(3, 3, 64, 23, gc=32)
upscale_model.load_state_dict(torch.load(esrgan_model_path), strict=True)
upscale_model.eval()
upscale_model = upscale_model.to(device)


@app.route('/txt2img', methods=['POST'])
def txt2img():
    prompt = request.form['prompt']
    out_file = request.form['outFile']
    seed = int(request.form['seed'])
    height = int(request.form['height'])
    width = int(request.form['width'])
    steps = int(request.form['steps'])

    generator = torch.Generator("cuda").manual_seed(seed)

    # run iterations and save output
    with autocast("cuda"):
        image = text_pipe(prompt, num_inference_steps=steps, generator=generator, height=height, width=width)["sample"][0][0]

        # embed metadata in exif
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text('parameters', f'Prompt: {prompt} Seed: {seed} Steps: {steps}')

        image.save(out_file, "PNG", pnginfo=pnginfo)

    return 'OK'


# @app.route('/img2img', methods=['POST'])
# def img2img():
#     prompt = request.form['prompt']
#     in_file = request.form['inFile']
#     out_file = request.form['outFile']
#     seed = int(request.form['seed'])
#     height = int(request.form['height'])
#     width = int(request.form['width'])
#     steps = int(request.form['steps'])

#     generator = torch.Generator("cuda").manual_seed(seed)

#     in_image = Image.open(in_file).convert("RGB")
#     in_image = in_image.resize((512, 512), resample=Image.Resampling.LANCZOS)

#     # run iterations and save output
#     with autocast("cuda"):
#         image = image_pipe(prompt, init_image=in_image, strength=0.75, num_inference_steps=steps, generator=generator, height=height, width=width)["sample"][0]
#         image.save(out_file, "PNG")

#     return 'OK'


@app.route('/upscale', methods=['POST'])
def upscale():
    in_file = request.form['inFile']
    out_file = request.form['outFile']

    # read image
    img = cv2.imread(in_file, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    # write upscaled image
    with torch.no_grad():
        output = upscale_model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    # downscale image so it fits on discord
    res = cv2.resize(output, dsize=(1536, 1536), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(out_file, res)

    return 'OK'