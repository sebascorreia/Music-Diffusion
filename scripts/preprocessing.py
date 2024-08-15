import os
import argparse
import io
import numpy as np
import logging
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Image, Value

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger("preprocessing")
import sys
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from music_diffusion.utils import Mel
from tqdm.auto import tqdm
def main(args):

    os.makedirs(args.output_dir, exist_ok=True)
    folder_path = args.audio_files
    audio_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith((".mp3", ".wav", ".m4a"))
    ]
    examples= []
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
                image=mel.audio_slice_to_image(slice)
                assert image.width==args.resolution and image.height==args.resolution
                if all(np.frombuffer(image.tobytes(), np.uint8)==255):
                    logger.warning("File %s slice %d is silent", audio_file, slice)
                    continue
                with io.BytesIO() as buffer:
                    image.save(buffer, "PNG")
                    bytes = buffer.getvalue()
                examples.extend(
                    [
                        {
                            "image":{"bytes":bytes},
                            "audio_file":audio_file,
                            "slice":slice,
                        }
                    ]
                )
    except Exception as e:
        print(e)
    finally:
        if len(examples)==0:
            logger.warning("No files found.")
            return
        ds =Dataset.from_pandas(
            pd.DataFrame(examples),
            features=Features(
                {
                    "image":Image(),
                    "audio_file":Value(dtype="string"),
                    "slice":Value(dtype="int16"),
                }
            ),
        )
        dsd = DatasetDict({"train": ds})
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