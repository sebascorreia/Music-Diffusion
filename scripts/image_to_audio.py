import os
import argparse
from PIL import Image
import sys
import IPython.display as display
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from music_diffusion.utils import Mel


def main(args):
    mel = Mel()

    if args.image is not None:
        image = Image.open(args.image)
        audio = mel.image_to_audio(image)
        display(image)
        mel.play_audio(audio)


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
