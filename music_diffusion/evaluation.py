from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os
import torch
from music_diffusion.utils import Mel

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
    for i,image in images:
        audio = mel.image_to_audio()
        mel.save_audio(audio, os.path.join(folder, f"samples{i}.wav"))
        image.save(os.path.join(folder, f"samples{i}.jpg"))

