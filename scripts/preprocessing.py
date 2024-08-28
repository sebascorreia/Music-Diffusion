import os
import argparse
import io
import numpy as np
import logging
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Image, Value, Audio
import soundfile as sf
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger("preprocessing")
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from music_diffusion.utils import Mel
from tqdm.auto import tqdm


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    folder_path = args.audio_files
    audio_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.endswith((".mp3", ".wav", ".m4a")):
                audio_files.append(os.path.join(root, f))
    examples = []
    mel = Mel(
        x_res=args.resolution[0],
        y_res=args.resolution[1],
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
    )
    try:
        for audio_file in tqdm(audio_files):
            try:
                mel.load_audio(audio_file)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(e)
                continue
            for slice in range(mel.get_number_of_slices()):
                audio_slice_data = mel.get_audio_slice(slice)
                with io.BytesIO() as audio_buffer:
                    sf.write(audio_buffer, audio_slice_data, args.sample_rate, format='WAV')
                    audio_slice_bytes = audio_buffer.getvalue()
                image = mel.audio_slice_to_image(slice)
                if args.label:
                    class_label = os.path.basename(os.path.dirname(audio_file))
                assert image.width == args.resolution[0] and image.height == args.resolution[1], "Wrong Resolution"
                if all(np.frombuffer(image.tobytes(), dtype=np.uint8) == 255):
                    logger.warning("File %s slice %d is silent", audio_file, slice)
                    continue
                with io.BytesIO() as buffer:
                    image.save(buffer, "PNG")
                    bytes = buffer.getvalue()
                if args.label:
                    examples.extend(
                        [
                            {
                                "image": {"bytes": bytes},
                                "audio_slice": {"bytes": audio_slice_bytes},
                                "audio_file": audio_file,
                                "slice": slice,
                                "label": class_label,
                            }
                        ]
                    )
                else:
                    examples.extend(
                        [
                            {
                                "image": {"bytes": bytes},
                                "audio_slice": {"bytes": audio_slice_bytes},
                                "audio_file": audio_file,
                                "slice": slice,
                            }
                        ]
                    )
    except Exception as e:
        print(e)
    finally:
        if len(examples) == 0:
            logger.warning("No files found.")
            return
        #image dataset
        if args.label:
            ds = Dataset.from_pandas(
                pd.DataFrame(examples),
                features=Features(
                    {
                        "image": Image(),
                        "audio_slice": Audio(sampling_rate=args.sample_rate),
                        "audio_file": Value(dtype="string"),
                        "slice": Value(dtype="int16"),
                        "label": Value(dtype="string"),
                    }
                ),
            )
        else:
            ds = Dataset.from_pandas(
                pd.DataFrame(examples),
                features=Features(
                    {
                        "image": Image(),
                        "audio_slice": Audio(sampling_rate=args.sample_rate),
                        "audio_file": Value(dtype="string"),
                        "slice": Value(dtype="int16"),
                    }
                ),
            )
        #Train/Test/ Splits
        train_test_split = ds.train_test_split(test_size=0.2, seed=42)
        test_split = train_test_split['test']

        dsd = DatasetDict({
            "train": train_test_split['train'],
            "test": test_split
        })
        dsd.save_to_disk(os.path.join(args.output_dir))
        if args.push_to_hub:
            dsd.push_to_hub(args.push_to_hub)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_files', type=str)
    parser.add_argument('--output_dir', type=str, default='data')
    parser.add_argument(
        "--resolution",
        type=str,
        default="256",
        help="Either square resolution or width,height.",
    )
    parser.add_argument("--label", type= bool, default=False)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--push_to_hub", type=str, default=None)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=2048)
    args = parser.parse_args()
    if args.audio_files is None:
        raise ValueError("Please provide audio file path")
    try:
        args.resolution = (int(args.resolution), int(args.resolution))
    except ValueError:
        try:
            args.resolution = tuple(int(x) for x in args.resolution.split(","))
            if len(args.resolution) != 2:
                raise ValueError
        except ValueError:
            raise ValueError("Resolution must be a tuple of two integers or a single integer.")
    assert isinstance(args.resolution, tuple)
    main(args)
