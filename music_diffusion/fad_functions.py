from fadtk import FrechetAudioDistance, ModelLoader
from typing import Callable, Union
from pathlib import Path
import numpy as np
import torchaudio
import torch
import multiprocessing
from tqdm import tqdm

def get_no_cache_embedding_path(model: str, audio_dir: Union[str, Path]) -> Path:
    """
    Get the path to the cached embedding npy file for an audio file.

    :param model: The name of the model
    :param audio_dir: The path to the audio file
    """
    audio_dir = Path(audio_dir)
    return audio_dir.parent / "embeddings" / model / audio_dir.with_suffix(".npy").name


def no_cache_embedding_files(files: Union[list[Path], str, Path], ml: ModelLoader, workers: int = 8,batch_size:int =50, **kwargs):
    """
    Get embeddings for all audio files in a directory.

    :param workers: basically 1/batch_size
    :param files: is files bruv
    :param ml_fn: A function that returns a ModelLoader instance.
    """
    if isinstance(files, (str, Path)):
        files = list(Path(files).glob('*.*'))

    # Filter out files that already have embeddings
    files = [f for f in files if not get_no_cache_embedding_path(ml.name, f).exists()]
    if len(files) == 0:
        print("All files already have embeddings, skipping.")
        return

    print(f"[Frechet Audio Distance] Loading {len(files)} audio files...")
    multiprocessing.set_start_method('spawn', force=True)

    # Determine the batch size based on the number of workers
    for i in tqdm(range(0, len(files), batch_size), desc="Processing batches"):
        chunk = files[i:i+batch_size]
        batches = list(np.array_split(chunk, workers))
            # Cache embeddings in parallel
        with torch.multiprocessing.Pool(workers) as pool:
            pool.map(no_cache_embedding_batch, [(b, ml, kwargs) for b in batches])


def no_cache_embedding_batch(args):
    fs: list[Path]
    ml: ModelLoader
    fs, ml, kwargs = args
    fad = NoCacheFAD(ml, **kwargs)
    for f in fs:
        print(f"Loading {f} using {ml.name}")
        fad.No_cache_embedding_file(f)


class NoCacheFAD(FrechetAudioDistance):

    def cache_embedding_file(self, audio_dir: Union[str, Path]):
        """
        Compute embedding for an audio file and cache it to a file.
        """
        emb_path = get_no_cache_embedding_path(self.ml.name, audio_dir)

        if emb_path.exists():
            return

        # Load file, get embedding, save embedding
        wav_data = self.load_audio(audio_dir)
        embd = self.ml.get_embedding(wav_data)
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, embd)

    def load_audio(self, f: Union[str, Path]):
        f = Path(f)
        save_path = f.parent / "convert" / str(self.ml.sr)
        new = (save_path / f.name).with_suffix(".wav")
        if not new.exists():
            save_path.mkdir(parents=True, exist_ok=True)
            x, fsorig = torchaudio.load(f)
            x = torch.mean(x, 0).unsqueeze(0)  # convert to mono
            resampler = torchaudio.transforms.Resample(
                fsorig,
                self.ml.sr,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492,
            )
            y = resampler(x)
            torchaudio.save(new, y, self.ml.sr, encoding="PCM_S", bits_per_sample=16)
            return self.ml.load_wav(new)
