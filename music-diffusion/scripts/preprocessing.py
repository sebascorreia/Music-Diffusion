import os
import re
import argparse

def main(args):

    os.makedirs(args.output_dir, exist_ok=True)
    audio_files =[
        os.path.join(root,file)
        for root, _, files in os.walk(args.audio_dir)
        for file in files
        if re.search("\.(mp3|wav|m4a)$", file, re.IGNORECASE)
    ]
    examples= []



if __name__ == '__main__':
    parser = argparse.ArgumentParser
    parser.add_argument('--audio_files', type=str)
    parser.add_argument('--output_dir', type=str, default='data')
    parser.add_argument("--resolution",type=str,default="256")
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--push_to_hub", type=str, default=None)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=2048)
    args = parser.parse_args()
    if args.audio_files is None:
        raise ValueError("Please provide audio file path")
    main(args)