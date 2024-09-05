import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from diffusers import DDPMScheduler, DDIMScheduler, DDPMPipeline, DDIMPipeline
from music_diffusion.models import ConditionalDDIMPipeline
import torch
from music_diffusion.evaluation import generate
from music_diffusion.utils import Mel

def main(args):
    if args.scheduler == "ddpm":
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.time_steps)  # linear b_t [0.0001,0.02]
        pipeline = DDPMPipeline.from_pretrained(args.from_pretrained, scheduler=noise_scheduler)
    else:
        noise_scheduler = DDIMScheduler(num_train_timesteps=args.time_steps)
        if args.cond:
            pipeline = ConditionalDDIMPipeline.from_pretrained(args.from_pretrained, scheduler=noise_scheduler)
        else:
            pipeline = DDIMPipeline.from_pretrained(args.from_pretrained, scheduler=noise_scheduler)
    if args.from_pretrained == "sebascorreia/DDPM-Maestro-full":
        mel = Mel()
    else:
        mel = Mel(x_res=128,
                  y_res=128,
                  hop_length=128,
                  sample_rate=16000,
                  n_fft=1024,
                  )
    pipeline.to(torch.device("cuda"))
    generate(args,pipeline,mel)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_pretrained', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./eval')
    parser.add_argument('--num_gen_img', type=int, default=2)
    parser.add_argument('--scheduler', type=str, default="ddim")
    parser.add_argument('--time_steps', type=int, default=50)
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument("--cond", type=bool, default=False)
    parser.add_argument('--class', type= str, default=None)
    args = parser.parse_args()
    if args.from_pretrained is None:
        raise ValueError("Please specify a pretrained model")
    main(args)