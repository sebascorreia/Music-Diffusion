from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os
import torch
from torchvision.datasets import folder
import subprocess

from music_diffusion.utils import Mel
import fadtk
from datasets import load_from_disk

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
    gen_images = pipeline(
        batch_size=1000,
        generator=torch.Generator(device='cpu').manual_seed(55),
        # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images
    gen_folder= os.path.join(args.output_dir, f"test{epoch}")
    for i,image in enumerate(gen_images):
        audio = mel.image_to_audio(image)
        mel.save_audio(audio, os.path.join(gen_folder, f"genaudio{i}.wav"))
    audio_set = real_dataset['audio_slice']
    real_folder = os.path.join(args.output_dir, "real_data")
    for i,audio in enumerate(audio_set):
        mel.save_audio(audio, os.path.join(real_folder, f"audio{i}.wav"))
    model = fadtk.CLAPLaionModel(type='music')
    fadtk.cache_embedding_files(real_folder, model, workers=8)
    fadtk.cache_embedding_files(gen_folder, model, workers=8)
    fad = fadtk.FrechetAudioDistance(model, audio_load_worker=8, load_model=False)
    score = fad.score(real_folder, gen_folder)
    inf_r2 = None
    return score










