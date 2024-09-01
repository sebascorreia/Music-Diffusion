import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from diffusers import DDPMScheduler, DDIMScheduler, DDPMPipeline, DDIMPipeline
import torch
from music_diffusion.evaluation import generate

def main(args):
    if args.scheduler == "ddpm":
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.time_steps)  # linear b_t [0.0001,0.02]
        pipeline = DDPMPipeline.from_pretrained(args.from_pretrained, scheduler=noise_scheduler)
    else:
        noise_scheduler = DDIMScheduler(num_train_timesteps=args.time_steps)
        pipeline = DDIMPipeline.from_pretrained(args.from_pretrained, scheduler=noise_scheduler, eta=args.eta)
    pipeline.to(torch.device("cuda"))
    generate(args,pipeline)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_pretrained', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./eval')
    parser.add_argument('--num_gen_img', type=int, default=2)
    parser.add_argument('--scheduler', type=str, default="ddim")
    parser.add_argument('--time_steps', type=int, default=50)
    parser.add_argument('--eta', type=float, default=0.0)
    args = parser.parse_args()
    if args.from_pretrained is None:
        raise ValueError("Please specify a pretrained model")
    main(args)