import os
import argparse
from PIL import Image
import sys
import IPython.display as display
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from music_diffusion.utils import Mel
from datasets import load_from_disk


def main(args):
    mel = Mel()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.image is not None:
        image = Image.open(args.image)
        audio = mel.image_to_audio(image)
        image.save(os.path.join(args.output_dir, "image.png"))
        mel.save_audio(audio, os.path.join(args.output_dir, 'audio.wav'))
    else:
        dataset = load_from_disk(args.dataset)
        dataset = dataset['train']
        for i,data in enumerate(dataset):
            spectrogram = data["image"]
            audio = mel.image_to_audio(spectrogram)
            mel.save_audio(audio, os.path.join(args.output_dir, f"audio_{i}.wav"))
            spectrogram.save(os.path.join(args.output_dir, f"spectrogram_{i}.jpg"))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()
    if args.dataset is None and args.image is None:
        raise ValueError('No input received')
    main(args)
