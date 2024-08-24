import argparse
import os
from diffusers import DDPMScheduler, DDIMScheduler, DDPMPipeline
from datasets import load_dataset, load_from_disk
from music_diffusion.evaluation import evaluate, FAD
import torch
def main(args):
    if args.scheduler == "ddpm":
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.train_steps)  # linear b_t [0.0001,0.02]
    else:
        noise_scheduler = DDIMScheduler(num_train_timesteps=args.train_steps)
    pipeline = DDPMPipeline.from_pretrained(args.from_pretrained, scheduler=noise_scheduler)
    pipeline.to(torch.device("cuda"))
    fad_score = FAD(args,pipeline)
    print("FAD score is: ", fad_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_pretrained', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./eval')
    parser.add_argument('--num_gen_img', type=int, default=500)
    parser.add_argument('--dataset', type=str, default="sebascorreia/Maestro20h")
    parser.add_argument('--scheduler', type=str, default="ddpm")

    args = parser.parse_args()
    main(args)