import argparse
import os
import sys
import torch
from diffusers import DDPMScheduler, DDIMScheduler, DDPMPipeline
from datasets import load_dataset, load_from_disk
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from music_diffusion.evaluation import interpolation
def main(args):
    if args.scheduler == "ddpm":
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)  # linear b_t [0.0001,0.02]
    else:
        noise_scheduler = DDIMScheduler(num_train_timesteps=50)
    pipeline = DDPMPipeline.from_pretrained(args.from_pretrained, scheduler=noise_scheduler)
    pipeline.to("cuda")
    inter_img = interpolation(img1, img2, pipeline= pipeline)
    print(type(inter_img))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', type=str, default=None)
    parser.add_argument('--img2', type=str, default=None)
    parser.add_argument('--scheduler', type=str, default='DDPM')
    parser.add_argument('--from_pretrained', type=str, default="sebascorreia/DDPM-maestro20h")
    parser.add_argument('--output_dir', type=str, default='./interpol')
    parser.add_argument('--intp_type',type=str,default='slerp')
    parser.add_argument('--lamb_val', type=float, default=0.5)
    parser.add_argument('--timesteps', type=int, default=50)
    args = parser.parse_args()
    if args.img1 is None or args.img2 is None:
        raise ValueError('Enter image paths to interpolate')
    main(args)
