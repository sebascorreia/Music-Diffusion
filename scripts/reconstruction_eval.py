import argparse
import os
import sys
from diffusers import DDPMScheduler, DDIMScheduler, DDPMPipeline, DDIMPipeline
from datasets import load_dataset, load_from_disk
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from music_diffusion.models import ConditionalDDIMPipeline
import multiprocessing
from music_diffusion.utils import Mel
from music_diffusion.evaluation import reconstruction
import torch
from tqdm import tqdm

label_mapping = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
                 'nine': 9, }
augmentations = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def main(args):
    accelerator = Accelerator()
    if os.path.exists(args.dataset):
        dataset = load_from_disk(args.dataset)["test"]
    else:
        try:
            dataset = load_dataset(args.dataset, split="test")
        except Exception as e:
            raise ValueError(f"Dataset error: {str(e)} ")

    def transform(examples):
        images = [augmentations(image) for image in examples["image"]]
        if args.conditional:
            labels = [label_mapping[label] for label in examples["label"]]
            return {"input": images, "label": labels}
        return {"input": images}

    dataset.set_transform(transform)

    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
    if args.from_pretrained == "sebascorreia/DDPM-sc09-conditional-2":
        pipeline = ConditionalDDIMPipeline.from_pretrained(args.from_pretrained, scheduler=noise_scheduler)
    else:
        pipeline = DDIMPipeline.from_pretrained(args.from_pretrained, scheduler=noise_scheduler)
    pipeline.to(accelerator.device)
    pipeline.unet.eval()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    pipeline, dataloader = accelerator.prepare(pipeline, dataloader)
    mean_mse = 0.0
    progress_bar = tqdm(total=len(dataloader))
    progress_bar.set_description("MSE Reconstruction")
    for step, batch in enumerate(dataloader):
        images = batch['input']
        images = torch.stack([img for img in images]).to(pipeline.device)
        if args.conditional:
            labels = batch['label']
            labels = torch.tensor([label for label in labels]).to(pipeline.device)
            print("LABEL TENSOR SHAPE: ", labels.shape)
        else:
            labels = None
        batch_mse = reconstruction(images, pipeline, args.timesteps, labels)
        mean_mse += batch_mse.mean().item()

        progress_bar.update(1)
    total_mse = mean_mse / len(dataloader)
    print(total_mse)
    os.makedirs(args.output_dir, exist_ok=True)
    output_file_path = os.path.join(args.output_dir, "MSE_score.txt")
    with open(output_file_path, "w") as f:
        f.write(f"MSE score is: {total_mse}\n")
    print("MSE score is: ", total_mse)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_pretrained', type=str, default="sebascorreia/DDPM-maestro20h")
    parser.add_argument('--output_dir', type=str, default='/content/eval')
    parser.add_argument('--dataset', type=str, default="sebascorreia/Maestro20h",
                        choices=["sebascorreia/sc09", "sebascorreia/Maestro20h"])
    parser.add_argument('--timesteps', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--conditional', type=bool, default=False)
    args = parser.parse_args()

    main(args)
