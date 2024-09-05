from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os
import torch
from torchvision.datasets import folder
import subprocess
from typing import Callable, Union
from pathlib import Path
from music_diffusion.utils import Mel
import fadtk
from datasets import load_from_disk, load_dataset
from .fad_functions import *
from pathlib import Path
from fadtk import ModelLoader
import numpy as np
import glob
from PIL import Image
import torchvision.transforms as transforms

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # Convert PIL image to tensor
])

def generate(args, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    mel = Mel()
    total_images = args.num_gen_img
    batch_size = 6
    image_count = 0
    folder = os.path.join(args.output_dir, f"eval")
    os.makedirs(folder, exist_ok=True)
    while image_count < total_images:
        remaining_images = total_images - image_count
        current_batch_size = min(batch_size, remaining_images)
        if args.cond:
            with torch.no_grad():
                gen_images = pipeline(
                    batch_size=current_batch_size,
                    generator=torch.Generator(device='cpu').manual_seed(55 + image_count),
                    eta=args.eta,
                    num_inference_steps=args.time_steps,
                    class_labels = torch.randint(0, 10, (batch_size,)).to("cuda")
                    # Use a separate torch generator to avoid rewinding the random state of the main training loop
                ).images
        else:
            with torch.no_grad():
                gen_images = pipeline(
                    batch_size=current_batch_size,
                    generator=torch.Generator(device='cpu').manual_seed(55 + image_count),
                    eta=args.eta,
                    num_inference_steps=args.time_steps,
                    # Use a separate torch generator to avoid rewinding the random state of the main training loop
                ).images
        image_count += current_batch_size

        for i,image in enumerate(gen_images):
            audio = mel.image_to_audio(image)
            mel.save_audio(audio, os.path.join(folder, f"samples{i}.wav"))
            image.save(os.path.join(folder, f"samples{i}.jpg"))

def denoise(noisy_img, pipeline, timestep):
    pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(noisy_img.device)
    for t in reversed(range(timestep)):
        t_tensor = torch.tensor([t], device=noisy_img.device)
        with torch.no_grad():
            model_output = pipeline.unet(noisy_img, t_tensor).sample
            denoisy_img = pipeline.scheduler.step(model_output, t_tensor, noisy_img).prev_sample
    return denoisy_img
def lerp(xt1,xt2,lamb):
    return (1-lamb) * xt1 + lamb * xt2
def slerp(xt1, xt2, lamb):
    #code based on: https://enzokro.dev/blog/posts/2022-11-16-pytorch-slerp/
    thr = 0.9995
    xt1norm = xt1 / torch.norm(xt1, dim=-1, keepdim=True )
    xt2norm = xt2 / torch.norm(xt2, dim=-1, keepdim=True )
    omg = (xt1norm * xt2norm).sum(dim=-1)
    if (torch.abs(omg) > thr).any():
        return lerp(xt1, xt2, lamb)
    else:
        theta = torch.acos(omg)
        s1 = torch.sin(theta - (theta * lamb))/ torch.sin(theta)
        s2 = torch.sin(theta * lamb)/torch.sin(theta)
        return (s1.unsqueeze(-1) * xt1) + (s2.unsqueeze(-1) * xt2)


def interpolation(img1,img2, pipeline, noise_timesteps=999, denoise_timesteps = 50,lamb=0.5, intp_type='slerp '):


    img1 = preprocess(img1).unsqueeze(0).to("cuda")
    img2 = preprocess(img2).unsqueeze(0).to("cuda")
    timesteps = torch.tensor([noise_timesteps], device=img1.device)
    noise1 = torch.randn_like(img1).to('cuda')
    noise2 = torch.randn_like(img2).to('cuda')
    xt1 = pipeline.scheduler.add_noise(img1, noise1, timesteps)
    xt2 = pipeline.scheduler.add_noise(img2, noise2, timesteps)

    if intp_type == 'linear':
        xt_bar = lerp(xt1, xt2, lamb)
    else:
        xt_bar = slerp(xt1, xt2, lamb)
    x0_bar = denoise(xt_bar,pipeline,denoise_timesteps)
    to_pil = transforms.ToPILImage()
    x0_bar = to_pil(x0_bar)
    return x0_bar
def reconstruction(img, pipeline,  timesteps = 50):
    original = preprocess(img).unsqueeze(0).to("cuda")
    noise = torch.randn_like(original)
    t = torch.tensor([timesteps], device=original.device)
    noisy_img = pipeline.scheduler.add_noise(original, noise, t)
    re_img = denoise(noisy_img, pipeline, timesteps)

    return re_img, mse(re_img, original)

def mse(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return mse
def FAD(args, pipeline):
    try:
        real_dataset = load_from_disk(args.dataset)[args.fad_split]
    except FileNotFoundError:
        try:
            real_dataset = load_dataset(args.dataset, split=args.fad_split)
        except Exception as e:
            raise e
    mel = Mel()
    gen_folder= os.path.join(args.output_dir, f"eval")
    os.makedirs(gen_folder, exist_ok=True)
    existing_audios = glob.glob(os.path.join(gen_folder, "*.wav"))
    total_images = args.num_gen_img - len(existing_audios)
    if total_images < 0:
        total_images = 0
    batch_size = 6
    image_count=0
    while image_count < total_images:
        remaining_images = total_images - image_count
        current_batch_size = min(batch_size, remaining_images)
        if args.cond:
            with torch.no_grad():
                gen_images = pipeline(
                    batch_size=current_batch_size,
                    generator=torch.Generator(device='cpu').manual_seed(55 + image_count),
                    eta=args.eta,
                    num_inference_steps=args.time_steps,
                    class_labels = torch.randint(0, 10, (batch_size,)).to("cuda")
                    # Use a separate torch generator to avoid rewinding the random state of the main training loop
                ).images
        else:
            with torch.no_grad():
                gen_images = pipeline(
                    batch_size=current_batch_size,
                    eta=args.eta,
                    num_inference_steps = args.time_steps,
                    generator=torch.Generator(device='cpu').manual_seed(55 + image_count),
            # Use a separate torch generator to avoid rewinding the random state of the main training loop
                ).images

        for i,image in enumerate(gen_images):
            audio = mel.image_to_audio(image)
            mel.save_audio(audio, os.path.join(gen_folder, f"genaudio{image_count + i}.wav"))
        image_count += current_batch_size
    audio_set = real_dataset['audio_slice']
    real_folder = os.path.join(args.output_dir, "real_data")
    os.makedirs(real_folder, exist_ok=True)
    existing_real_audios = glob.glob(os.path.join(real_folder, "*.wav"))
    if len(audio_set) > len(existing_real_audios):
        for i,audio in enumerate(audio_set):
            if i > len(existing_real_audios):
                mel.save_audio(audio, os.path.join(real_folder, f"audio{i}.wav"))
    if args.fad_model=='enc24':
        model = fadtk.EncodecEmbModel(variant='24k')
    elif args.fad_model=='enc48':
        model = fadtk.EncodecEmbModel(variant='48k')
    elif args.fad_model=='vgg':
        model = fadtk.VGGishModel()
    elif args.fad_model=='dac':
        model = fadtk.DACModel()
    else:
        raise ValueError("fadtk model not implemented")
    print("model: ", model.name)
    no_cache_embedding_files(real_folder, model, workers=3, batch_size=6)
    print("Real folder embeddings done")
    no_cache_embedding_files(gen_folder, model, workers=3, batch_size=6)
    print("Generated folder embeddings done")
    fad = NoCacheFAD(model, audio_load_worker=16, load_model=False)
    print("FAD COMPUTED")
    score = fad.score(real_folder, gen_folder)
    print("FAD Score:", score)
    return score















