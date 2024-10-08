from diffusers import DDPMPipeline, DDIMScheduler
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
from music_diffusion.models import ConditionalDDIMPipeline
label_mapping = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
}
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # Convert PIL image to tensor
])
def postprocess(tensor):
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    tensor = (tensor/2 + 0.5).clamp(0, 1)
    tensor = tensor.cpu().permute(0,2,3,1).numpy()
    tensor = (tensor*255).round().astype("uint8")
    pil_images = [
        Image.fromarray(image[:, :, 0]) if image.shape[2] == 1
        else Image.fromarray(image,mode="RGB".convert("L"))
        for image in tensor
    ]
    if len(pil_images) == 1:
        return pil_images[0]
    return pil_images
def generate(args, pipeline,mel):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`

    total_images = args.num_gen_img
    batch_size = 6
    image_count = 0
    folder = os.path.join(args.output_dir, f"eval")
    os.makedirs(folder, exist_ok=True)
    while image_count < total_images:
        remaining_images = total_images - image_count
        current_batch_size = min(batch_size, remaining_images)
        if args.cond:
            if args.label== None:
                label = torch.randint(0, 10, (current_batch_size,)).to("cuda")
            else:
                label = label_mapping[args.label]
                label = torch.full((current_batch_size,), label).to("cuda")
            with torch.no_grad():
                gen_images = pipeline(
                    batch_size=current_batch_size,
                    generator=torch.Generator(device='cpu').manual_seed(args.seed),
                    eta=args.eta,
                    num_inference_steps=args.time_steps,
                    class_labels = label
                    # Use a separate torch generator to avoid rewinding the random state of the main training loop
                ).images
        else:
            with torch.no_grad():
                gen_images = pipeline(
                    batch_size=current_batch_size,
                    generator=torch.Generator(device='cpu').manual_seed(args.seed),
                    eta=args.eta,
                    num_inference_steps=args.time_steps,
                    # Use a separate torch generator to avoid rewinding the random state of the main training loop
                ).images
        image_count += current_batch_size

        for i,image in enumerate(gen_images):
            audio = mel.image_to_audio(image)
            mel.save_audio(audio, os.path.join(folder, f"samples{i}.wav"))
            image.save(os.path.join(folder, f"samples{i}.jpg"))
def noise(sample, pipeline, timesteps, label=None): #taken from teticio/audio-diffusion
    assert isinstance(pipeline.scheduler, DDIMScheduler)
    pipeline.scheduler.set_timesteps(timesteps)
    for t in torch.flip(pipeline.scheduler.timesteps, (0,)):
        prev_t = t-pipeline.scheduler.config.num_train_timesteps // pipeline.scheduler.num_inference_steps
        alpha_prod_t = pipeline.scheduler.alphas_cumprod[t]
        prev_alpha_prod_t = (
            pipeline.scheduler.alphas_cumprod[prev_t] if prev_t >=0
            else pipeline.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        t= t.to("cuda")
        sample = sample.to("cuda")
        with torch.no_grad():
            if label is not None:
                label = torch.tensor(label).to("cuda")
                pred_noise = pipeline.unet(sample, t,label)["sample"]
            else:
                pred_noise = pipeline.unet(sample, t)["sample"]
        pred_sample_dir = (1-prev_alpha_prod_t )**(0.5) * pred_noise
        sample = (sample - pred_sample_dir) * (prev_alpha_prod_t **(-0.5))
        sample = (sample * (alpha_prod_t ** (0.5))) + ((beta_prod_t**(0.5)) * pred_noise)
    return sample

def denoise(noisy_img, pipeline, timesteps,label=None):
    pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(noisy_img.device)
    pipeline.scheduler.set_timesteps(timesteps)
    noisy_img = noisy_img
    with torch.no_grad():
      for step, t in enumerate(pipeline.scheduler.timesteps[0:]):
          if label is not None:
              label = torch.tensor(label).to("cuda")
              model_output = pipeline.unet(noisy_img, t, label)["sample"]
          else:
              model_output = pipeline.unet(noisy_img, t)["sample"]

          noisy_img = pipeline.scheduler.step(model_output=model_output,
                                                    timestep=t,
                                                    sample=noisy_img)["prev_sample"]

    return noisy_img
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


def interpolation(img1,img2, pipeline, timesteps = 50,lamb=0.5, intp_type='slerp ',class1=None, class2=None,  interclass= None):


    img1 = preprocess(img1).unsqueeze(0).to("cuda")
    img2 = preprocess(img2).unsqueeze(0).to("cuda")

    xt1 = noise(img1, pipeline, timesteps, class1)
    xt2 = noise(img2, pipeline, timesteps,class2)

    if intp_type == 'linear':
        xt_bar = lerp(xt1, xt2, lamb)
    else:
        xt_bar = slerp(xt1, xt2, lamb)
    x0_bar = denoise(xt_bar,pipeline,timesteps, interclass)
    inter_img = postprocess(x0_bar)
    return inter_img
def reconstruction(img, pipeline,  timesteps = 50, label= None):
    eval = False
    if isinstance(img, torch.Tensor):
        eval = True
    else:
        img = preprocess(img).unsqueeze(0).to("cuda")
    noisy_img = noise(img, pipeline, timesteps, label)
    re_img = denoise(noisy_img, pipeline, timesteps,label)
    if eval:
        return mse(re_img, img)

    return re_img, mse(re_img, img)

def mse(img1, img2):
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    return mse

def get_available_folder(directory,name, max_files):
    folder_index = 0
    while True:
        f_name = (folder_index * max_files) +1
        current_folder = os.path.join(directory, f"{name}{f_name}")
        os.makedirs(current_folder, exist_ok=True)
        existing_audios = glob.glob(os.path.join(current_folder, "*.wav"))
        if len(existing_audios) < max_files:
            return current_folder, len(existing_audios)
        folder_index += 1

def FAD(args, mel, pipeline):
    try:
        real_dataset = load_from_disk(args.dataset)[args.fad_split]
    except FileNotFoundError:
        try:
            real_dataset = load_dataset(args.dataset, split=args.fad_split)
        except Exception as e:
            raise e
    if args.cond:
        p_gen_folder = os.path.join(args.output_dir, "cond")
        gen_folder = os.path.join(p_gen_folder, "genaudio")
    else:
        p_gen_folder = os.path.join(args.output_dir, "uncond")
        gen_folder = os.path.join(p_gen_folder, "genaudio")
    os.makedirs(p_gen_folder,exist_ok=True)
    os.makedirs(gen_folder, exist_ok=True)
    existing_audios = glob.glob(os.path.join(gen_folder,'**', "*.wav"), recursive=True)
    total_images = args.num_gen_img - len(existing_audios)
    if total_images < 0:
        total_images = 0
    batch_size = args.gen_batch_size
    image_count=0
    while image_count < total_images:
        remaining_images = total_images - image_count
        print("Remaining Images: ", remaining_images)
        current_batch_size = min(batch_size, remaining_images)
        if args.cond:
            with torch.no_grad():
                gen_images = pipeline(
                    batch_size=current_batch_size,
                    generator=torch.Generator(device='cpu').manual_seed(55 + image_count),
                    eta=args.eta,
                    num_inference_steps=args.time_steps,
                    class_labels = torch.randint(0, 10, (current_batch_size,)).to("cuda")
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
        current_folder, folder_count = get_available_folder(gen_folder, "genaudio",args.folder_max)
        for i,image in enumerate(gen_images):
            if folder_count > args.folder_max:
                current_folder, folder_count = get_available_folder(gen_folder, "genaudio", args.folder_max)
            audio = mel.image_to_audio(image)
            mel.save_audio(audio, os.path.join(current_folder, f"genaudio{image_count + i}.wav"))
            folder_count += 1
        image_count += current_batch_size
    audio_set = real_dataset['audio_slice']
    real_folder = os.path.join(args.output_dir, "Audio")
    os.makedirs(real_folder, exist_ok=True)
    existing_real_audios = glob.glob(os.path.join(real_folder,'**', "*.wav"), recursive=True)

    if len(audio_set) > len(existing_real_audios):
        current_folder, folder_count = get_available_folder(real_folder, "audio", args.folder_max)
        print("Current Folder:", str(current_folder))
        for i,audio in enumerate(audio_set):
            if i > len(existing_real_audios):
                if folder_count > args.folder_max:
                    current_folder, folder_count = get_available_folder(real_folder, "audio", args.folder_max)
                    print("Current Folder:", str(current_folder))
                mel.save_audio(audio, os.path.join(current_folder, f"audio{i}.wav"))
                folder_count += 1
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
    real_embs = os.path.join(args.output_dir, f"{model.name}/Embeddings")
    no_cache_embedding_files(real_folder, embs=real_embs,ml=model, workers=32, batch_size=32)
    print("Real folder embeddings done")
    gen_emb = os.path.join(p_gen_folder, f"{model.name}/Embeddings")
    no_cache_embedding_files(gen_folder,embs=gen_emb, ml=model, workers=32, batch_size=32)
    print("Generated folder embeddings done")
    fad = NoCacheFAD(model, audio_load_worker=16, load_model=False)
    print("FAD COMPUTED")
    score = fad.score(real_folder, gen_folder)
    print("FAD Score:", score)
    return score















