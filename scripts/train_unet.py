import argparse
import os
import torch
import logging
from diffusers.training_utils import EMAModel
from datasets import load_from_disk, load_dataset
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, DDPMPipeline, DDIMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
import torchvision.transforms as transforms
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import torch.nn.functional as F
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from music_diffusion.evaluation import generate, FAD
from music_diffusion.models import Unet2d, CondUnet2d
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

def main(args):
    if os.path.exists(args.dataset):
        dataset = load_from_disk(args.dataset)["train"]
    else:
        try:
            dataset = load_dataset(args.dataset, split="train")
        except Exception as e:
            raise ValueError(f"Dataset error: {str(e)} ")

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.train_steps)  #linear b_t [0.0001,0.02]
    if args.from_pretrained is not None:
        pipeline = DDPMPipeline.from_pretrained(args.from_pretrained, scheduler=noise_scheduler)
        model = pipeline.unet
    else:
        if args.conditional:
            model =CondUnet2d(dataset['image'][0].width, args.classes)
        else:
            model = Unet2d(dataset['image'][0].width)  # Default diffusion Unet 2d model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    augmentations = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform(examples):
        images = [augmentations(image) for image in examples["image"]]
        if args.conditional:
            labels = [label_mapping[label] for label in examples["label"]]
            return {"input": images, "label": labels}
        return {"input": images}

    dataset.set_transform(transform)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
    device = next(model.parameters()).device
    ema_model = EMAModel(
        getattr(model, "module", model).parameters(),
        use_ema_warmup=args.ema_warmup,
        power=args.ema_power,
        max_value=args.ema_max_decay,
    )
    ema_model.to(device)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.epochs)
    )
    train_loop(args, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, ema_model)


def train_loop(args, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, ema_model):
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

        if epoch < args.start_epoch:
            for step in range(len(train_dataloader)):
                optimizer.step()
                lr_scheduler.step()
                global_step += 1
            if epoch == args.start_epoch - 1 and args.use_ema:
                ema_model.optimization_step = global_step
            continue
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        model.train()
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["input"]
            if args.conditional:
                class_labels = batch["label"]
            #sampling noise
            noise=torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]
            #sampling random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            ).long()
            #add noise to clean images

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                if args.conditional:
                    noise_pred = model(noisy_images, timesteps,class_labels=class_labels)["sample"]
                else:
                    noise_pred = model(noisy_images, timesteps)["sample"]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                if args.use_ema:
                    ema_model.step(model.parameters())
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step, }
            if args.use_ema:
                logs["ema_decay"] = ema_model.decay
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        progress_bar.close()
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            unet = accelerator.unwrap_model(model)
            if args.scheduler == 'ddpm':
                pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
            else:
                pipeline = DDIMPipeline(unet=unet, scheduler=noise_scheduler)
            if args.use_ema:
                ema_model.copy_to(unet.parameters())
            if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.epochs - 1:
                pipeline.save_pretrained(args.output_dir)
                if args.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=args.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"]
                    )

                else:
                    pipeline.save.pretrained(args.output_dir)
            if (epoch + 1) % args.save_image_epochs == 0:
                model.eval()
                generate(args, pipeline)
                model.train()
            if (epoch + 1) % args.fad == 0 or epoch == args.epochs - 1:
                fad_score = FAD(args, pipeline)
                print("FAD score is: ", fad_score)
                torch.set_grad_enabled(True)  # Re-enable gradient calculation
                model.train()


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
    parser.add_argument('--mixed_precision', type=str, default='no')
    parser.add_argument('--push_to_hub', type=bool, default=False)
    parser.add_argument('--hub_model_id', type=str, default=None)
    parser.add_argument('--save_image_epochs', type=int, default=10)
    parser.add_argument('--save_model_epochs', type=int, default=30)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--fad", type=int, default=1)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--ema_warmup", type=bool, default=False) #turn on in case of long training
    parser.add_argument("--ema_power", type=float, default=0.75) #use 2/3 to train for more than 1M steps
    parser.add_argument("--ema_max_decay", type=float, default=0.9999)
    parser.add_argument('--num_gen_img', type=int, default=500)
    parser.add_argument('--fad_split', type=str, default='test')
    parser.add_argument('--conditional', type=bool, default=False)
    parser.add_argument('--classes', type=int, default=10)
    parser.add_argument('--fad_model',
                        type=str, default='dac',
                        choices=['dac', 'enc24','enc48','vgg'],
                        help=("Choose FAD model: dac, enc24,enc48 or vgg"),)
    args = parser.parse_args()
    if args.dataset == None:
        raise ValueError("Please provide training data directory")
    if args.output_dir == None:
        raise ValueError("Please provide output directory")
    main(args)
