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
from datasets import load_from_disk
from .fad_functions import *
from pathlib import Path
from fadtk import ModelLoader
import numpy as np

def evaluate(args, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    mel = Mel()
    images = pipeline(
        batch_size=args.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(55), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images
    folder = os.path.join(args.output_dir, f"eval{epoch}")
    os.makedirs(folder, exist_ok=True)
    for i,image in enumerate(images):
        audio = mel.image_to_audio(image)
        mel.save_audio(audio, os.path.join(folder, f"samples{i}.wav"))
        image.save(os.path.join(folder, f"samples{i}.jpg"))

def FAD(args, epoch, pipeline):
    real_dataset = load_from_disk(args.dataset)["test"]
    mel = Mel()
    gen_folder= os.path.join(args.output_dir, f"test{epoch}")
    os.makedirs(gen_folder, exist_ok=True)
    total_images = 6
    batch_size = 6
    image_count=0
    while image_count < total_images:
        remaining_images = total_images - image_count
        current_batch_size = min(batch_size, remaining_images)
        gen_images = pipeline(
            batch_size=current_batch_size,
            generator=torch.Generator(device='cpu').manual_seed(55 + image_count),
        # Use a separate torch generator to avoid rewinding the random state of the main training loop
        ).images

        for i,image in enumerate(gen_images):
            audio = mel.image_to_audio(image)
            mel.save_audio(audio, os.path.join(gen_folder, f"genaudio{i}.wav"))
        image_count += current_batch_size
    audio_set = real_dataset['audio_slice']
    real_folder = os.path.join(args.output_dir, "real_data")
    os.makedirs(real_folder, exist_ok=True)
    for i,audio in enumerate(audio_set):
        mel.save_audio(audio, os.path.join(real_folder, f"audio{i}.wav"))
    model = fadtk.VGGishModel()
    print("model: ", model.name)
    no_cache_embedding_files(real_folder, model, workers=16)
    print("Real folder embeddings done")
    no_cache_embedding_files(gen_folder, model, workers=2)
    print("Generated folder embeddings done")
    fad = fadtk.FrechetAudioDistance(model, audio_load_worker=16, load_model=False)
    print("FAD COMPUTED")
    score = fad.score(real_folder, gen_folder)
    print("FAD Score:", score)
    return score















