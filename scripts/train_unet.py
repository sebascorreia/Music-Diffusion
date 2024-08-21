import argparse
import os
import torch
import logging
from datasets import load_from_disk, load_dataset
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
import torchvision.transforms as transforms
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import torch.nn.functional as F
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from music_diffusion.evaluation import evaluate, FAD
from music_diffusion.models import Unet2d


def main(args):
    if os.path.exists(args.dataset):
        dataset = load_from_disk(args.dataset)["train"]
    else:
        try:
            dataset = load_dataset(args.dataset, split="train")
        except Exception as e:
            raise ValueError(f"Dataset error: {str(e)} ")

    if args.scheduler == "ddpm":
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.train_steps)  #linear b_t [0.0001,0.02]
    else:
        noise_scheduler = DDIMScheduler(num_train_timesteps=args.train_steps)
    if args.from_pretrained is not None:
        pipeline = DDPMPipeline.from_pretrained(args.from_pretrained, scheduler=noise_scheduler)
        model = pipeline.unet
    else:
        model = Unet2d()  #Default diffusion Unet 2d model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    augmentations = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform(examples):
        images = [augmentations(image) for image in examples["image"]]
        return {"input": images}

    dataset.set_transform(transform)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.epochs)
    )
    train_loop(args, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)


def train_loop(args, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.grad_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(args.output_dir, "tensorboard"),
    )
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

    global_step = 0

    for epoch in range(args.epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        if epoch < args.start_epoch:
            for step in range(len(train_dataloader)):
                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)
                global_step += 1

        model.train()
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["input"]
            #sampling noise
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]
            #sampling random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            ).long()
            #add noise to clean images
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps)["sample"]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step, }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        progress_bar.close()
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            if (epoch + 1) % args.save_image_epochs == 0 or epoch == args.epochs - 1:
                evaluate(args, epoch, pipeline)

            if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.epochs - 1:
                if args.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=args.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"]
                    )
                else:
                    pipeline.save.pretrained(args.output_dir)
            if (epoch + 1) % args.fad == 0 or epoch == args.epochs - 1:
                fad_score = FAD(args, epoch, pipeline)
                print("FAD score is: ", fad_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--from_pretrained', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./model')
    parser.add_argument('--grad_accumulation_steps', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--scheduler', type=str, default='ddpm')
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument('--lr_warmup_steps', type=int, default=500)
    parser.add_argument('--mixed_precision', type=str, default='fp16')
    parser.add_argument('--push_to_hub', type=bool, default=False)
    parser.add_argument('--hub_model_id', type=str, default=None)
    parser.add_argument('--save_image_epochs', type=int, default=10)
    parser.add_argument('--save_model_epochs', type=int, default=30)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--fad", type=int, default=1)

    args = parser.parse_args()
    if args.dataset == None:
        raise ValueError("Please provide training data directory")
    if args.output_dir == None:
        raise ValueError("Please provide output directory")
    main(args)
