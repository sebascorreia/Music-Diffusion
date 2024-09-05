import argparse
import os
import sys
import torch
from diffusers import DDPMScheduler, DDIMScheduler, DDPMPipeline, DDIMPipeline
from datasets import load_dataset, load_from_disk

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from music_diffusion.evaluation import reconstruction
from music_diffusion.utils import Mel
from music_diffusion.models import ConditionalDDIMPipeline
from scripts.interpolate import get_img, save
import random

label_mapping = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
                 'seven': 7, 'eight': 8, 'nine': 9, }


def main(args):
    if args.dataset == 'sebascorreia/sc09':
        mel = Mel(x_res=128,
                  y_res=128,
                  hop_length=128,
                  sample_rate=16000,
                  n_fft=1024,
                  )
    else:
        mel = Mel()
    if os.path.exists(args.dataset):
        dataset = load_from_disk(args.dataset)["test"]
    else:
        try:
            dataset = load_dataset(args.dataset, split="test")
        except Exception as e:
            raise ValueError(f"Dataset error: {str(e)} ")
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
    if args.from_pretrained == "sebascorreia/DDPM-sc09-conditional-2":
        pipeline = ConditionalDDIMPipeline.from_pretrained(args.from_pretrained, scheduler=noise_scheduler)
    else:
        pipeline = DDIMPipeline.from_pretrained(args.from_pretrained, scheduler=noise_scheduler)
    pipeline.scheduler.set_timesteps(50, "cuda")
    if args.filter is not None:
        c1dataset = dataset.filter(lambda x: x["label"] == args.filter)
        img = get_img(args.img, c1dataset)
    else:
        img = get_img(args.img, dataset)
    if isinstance(pipeline, ConditionalDDIMPipeline):
        label = label_mapping[args.label]
    else:
        label = None
    re_img, mse = reconstruction(img, pipeline, args.timesteps, label)
    print("MSE RESULTS: ", mse)
    save(args, img, mel, "original")
    save(args, re_img, mel, "reconstruction")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=int, default=None, )
    parser.add_argument('--filter', type=str, default=None)
    parser.add_argument('--label', type=str, default=None)
    parser.add_argument('--from_pretrained', type=str, default="sebascorreia/DDPM-maestro20h")
    parser.add_argument('--output_dir', type=str, default='/content/reconstruct')
    parser.add_argument('--timesteps', type=int, default=50)
    parser.add_argument('--dataset', type=str, default="sebascorreia/sc09",
                        choices=["sebascorreia/sc09", "sebascorreia/Maestro20h"])
    args = parser.parse_args()

    main(args)
