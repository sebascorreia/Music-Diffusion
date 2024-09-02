import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from diffusers import DDPMScheduler, DDIMScheduler, DDPMPipeline, DDIMPipeline
from datasets import load_dataset, load_from_disk
from music_diffusion.evaluation import FAD
import torch


def main(args):
    if args.scheduler == "ddpm":
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)  # linear b_t [0.0001,0.02]
        pipeline = DDPMPipeline.from_pretrained(args.from_pretrained, scheduler=noise_scheduler)
    else:
        noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
        pipeline = DDIMPipeline.from_pretrained(args.from_pretrained, scheduler=noise_scheduler, eta=args.eta)
    pipeline.to(torch.device("cuda"))
    fad_score = FAD(args, pipeline)
    output_file_path = os.path.join(args.output_dir, "fad_score.txt")
    with open(output_file_path, "w") as f:
        f.write(f"FAD score is: {fad_score}\n")
    print("FAD score is: ", fad_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_pretrained', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./eval')
    parser.add_argument('--num_gen_img', type=int, default=1000)
    parser.add_argument('--dataset', type=str, default="sebascorreia/Maestro20h")
    parser.add_argument('--scheduler', type=str, default="ddim")
    parser.add_argument('--time_steps', type=int, default=50)
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--fad_split', type=str, default='train')
    parser.add_argument('--fad_model',
                        type=str, default='dac',
                        choices=['dac', 'enc24', 'enc48', 'vgg'],
                        help=("Choose FAD model: dac, enc24,enc48 or vgg"), )
    args = parser.parse_args()
    if args.from_pretrained is None:
        raise ValueError("Please specify a pretrained model")

    main(args)
