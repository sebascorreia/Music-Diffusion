import argparse
import os
import sys
import torch
from diffusers import DDPMScheduler, DDIMScheduler, DDPMPipeline, DDIMPipeline
from datasets import load_dataset, load_from_disk

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from music_diffusion.evaluation import interpolation
from music_diffusion.utils import Mel
from music_diffusion.models import ConditionalDDIMPipeline
import random

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


def save(args, mel, img, file):
    os.makedirs(args.output_dir, exist_ok=True)
    audio1 = mel.image_to_audio(img)
    mel.save_audio(audio1, os.path.join(args.output_dir, f"{file}.wav"))
    img.save(os.path.join(args.output_dir, f"{file}.jpg"))
    print(f"Saved {file}")


def get_img(n, dataset):
    if n is None:
        rand1 = random.randint(0, len(dataset) - 1)
        img = dataset['image'][rand1]
    else:
        img = dataset['image'][n]
    return img


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

    if args.class1 is not None:
        c1dataset = dataset.filter(lambda x: x["label"] == args.class1)
        img1 = get_img(args.img1, c1dataset)
    else:
        img1 = get_img(args.img1, dataset)
    if args.class2 is not None:
        c2dataset = dataset.filter(lambda x: x["label"] == args.class2)
        img2 = get_img(args.img2, c2dataset)
    else:
        img2 = get_img(args.img2, dataset)

    if args.scheduler == "ddpm":
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)  # linear b_t [0.0001,0.02]
        pipeline = DDPMPipeline.from_pretrained(args.from_pretrained, scheduler=noise_scheduler)
    else:
        noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
        if args.from_pretrained == "sebascorreia/DDPM-sc09-conditional-2":
            pipeline = ConditionalDDIMPipeline.from_pretrained(args.from_pretrained, scheduler=noise_scheduler)
        else:
            pipeline = DDIMPipeline.from_pretrained(args.from_pretrained, scheduler=noise_scheduler)
            pipeline.scheduler.set_timesteps(50, "cuda")
    pipeline.to("cuda")

    if isinstance(pipeline, ConditionalDDIMPipeline):
        try:
            class1 = label_mapping[args.class1]
            class2 = label_mapping[args.class2]
            interclass = label_mapping[args.interclass]
        except Exception as e:
            raise ("Missing Label: ", str(e))
    else:
        class1 = None
        class2 = None
        interclass = None

    inter_img = interpolation(
        img1=img1,
        img2=img2,
        pipeline=pipeline,
        timesteps=args.timesteps,
        lamb=args.lamb,
        intp_type=args.intp_type,
        class1=class1,
        class2=class2,
        interclass=interclass,
    )
    save(args, mel, img1, args.output_dir)
    save(args, mel, img2, args.output_dir)
    save(args, mel, inter_img, args.output_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', type=int, default=None, )
    parser.add_argument('--img2', type=int, default=None, )
    parser.add_argument('--class1', type=str, default=None)
    parser.add_argument('--class2', type=str, default=None)
    parser.add_argument('--interclass', type=str, default=None)
    parser.add_argument('--scheduler', type=str, default='DDPM')
    parser.add_argument('--from_pretrained', type=str, default="sebascorreia/DDPM-maestro20h")
    parser.add_argument('--output_dir', type=str, default='/interpol')
    parser.add_argument('--intp_type', type=str, default='slerp')
    parser.add_argument('--lamb', type=float, default=0.5)
    parser.add_argument('--timesteps', type=int, default=50)
    parser.add_argument('--dataset', type=str, default="sebascorreia/sc09",
                        choices=["sebascorreia/sc09", "sebascorreia/Maestro20h"])
    args = parser.parse_args()

    main(args)
