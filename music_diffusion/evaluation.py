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

def evaluate(args, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    mel = Mel()
    total_images = 100
    batch_size = 6
    image_count = 0

    while image_count < total_images:
        remaining_images = total_images - image_count
        current_batch_size = min(batch_size, remaining_images)

        with torch.no_grad():
            gen_images = pipeline(
                batch_size=current_batch_size,
                generator=torch.Generator(device='cpu').manual_seed(55 + image_count),
                # Use a separate torch generator to avoid rewinding the random state of the main training loop
            ).images
    folder = os.path.join(args.output_dir, f"eval{epoch}")
    os.makedirs(folder, exist_ok=True)
    for i,image in enumerate(images):
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


def interpolation(args, pipeline):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),        # Convert PIL image to tensor
    ])

    img1 = preprocess(Image.open(args.img1)).unsqueeze(0).to("cuda")
    img2 = preprocess(Image.open(args.img2)).unsqueeze(0).to("cuda")
    timesteps = args.timesteps
    noise1 = torch.randn_like(img1)
    noise2 = torch.randn_like(img2)
    xt1 = pipeline.scheduler.add_noise(img1, noise1, timesteps)
    xt2 = pipeline.scheduler.add_noise(img2, noise2, timesteps)

    if args.intp_type == 'linear':
        xt_bar = lerp(xt1, xt2, args.lambda_val)
    else:
        xt_bar = slerp(xt1, xt2, args.lambda_val)
    x0_bar = denoise(xt_bar,pipeline,timesteps)
    return x0_bar


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
        with torch.no_grad():
            gen_images = pipeline(
                batch_size=current_batch_size,
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
    for i,audio in enumerate(audio_set):
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















