from fadtk import FrechetAudioDistance, ModelLoader
from typing import Callable, Union
from pathlib import Path
import numpy as np
import torchaudio
import torch
import multiprocessing
import fadtk
import gc
from tqdm import tqdm
from scipy import linalg
import signal
PathLike = Union[str, Path]
from hypy_utils.nlp_utils import substr_between
from hypy_utils.tqdm_utils import pmap



def _process_file(file: PathLike):
    try:
        embd = np.load(file)
        n = embd.shape[0]
        mu = np.mean(embd, axis=0)
        cov = np.cov(embd, rowvar=False) * (n - 1)
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None, None, 0
    return mu,cov,n


def calculate_embd_statistics_online(files: list[PathLike], chunk_size: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mean and covariance matrix of a list of embeddings in an online manner, processing files in chunks.

    :param files: A list of npy files containing ndarrays with shape (n_frames, n_features)
    :param chunk_size: Number of files to process in each chunk
    """
    assert len(files) > 0, "No files provided"

    # Load the first file to get the embedding dimension
    embd_dim = np.load(files[0]).shape[-1]

    # Initialize the mean and covariance matrix
    mu = np.zeros(embd_dim)
    S = np.zeros((embd_dim, embd_dim))  # Sum of squares for online covariance computation
    n = 0  # Counter for total number of frames
    total_chunks = (len(files) + chunk_size - 1) // chunk_size
    # Process the files in chunks
    for i in range(0, len(files), chunk_size):
        chunk = files[i:i+chunk_size]
        current_chunk = i // chunk_size + 1
        results = pmap(_process_file, chunk, desc=f'Calculating statistics for chunk {current_chunk}/{total_chunks}')
        for _mu, _S, _n in results:
            if _mu is None or _S is None or _n==0:
                continue
            delta = _mu - mu
            mu += _n / (n + _n) * delta
            S += _S + delta[:, None] * delta[None, :] * n * _n / (n + _n)
            n += _n

    if n < 2:
        return mu, np.zeros_like(S)
    else:
        cov = S / (n - 1)  # compute the covariance matrix
        return mu, cov



def calc_frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6):
    """
    Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
            inception net (like returned by the function 'get_predictions')
            for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
            representative data set.
    -- cov1: The covariance matrix over activations for generated samples.
    -- cov2: The covariance matrix over activations, precalculated on an
            representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    cov1 = np.atleast_2d(cov1)
    cov2 = np.atleast_2d(cov2)

    assert mu1.shape == mu2.shape, \
        f'Training and test mean vectors have different lengths ({mu1.shape} vs {mu2.shape})'
    assert cov1.shape == cov2.shape, \
        f'Training and test covariances have different dimensions ({cov1.shape} vs {cov2.shape})'

    diff = mu1 - mu2

    # Product might be almost singular
    # NOTE: issues with sqrtm for newer scipy versions
    # using eigenvalue method as workaround
    covmean_sqrtm, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)

    # eigenvalue method
    D, V = linalg.eig(cov1.dot(cov2))
    covmean = (V * np.sqrt(D)) @ linalg.inv(V)

    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(cov1.shape[0]) * eps
        covmean = linalg.sqrtm((cov1 + offset).dot(cov2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    tr_covmean_sqrtm = np.trace(covmean_sqrtm)
    if np.iscomplexobj(tr_covmean_sqrtm):
        if np.abs(tr_covmean_sqrtm.imag) < 1e-3:
            tr_covmean_sqrtm = tr_covmean_sqrtm.real

    if not (np.iscomplexobj(tr_covmean_sqrtm)):
        delt = np.abs(tr_covmean - tr_covmean_sqrtm)
        if delt > 1e-3:
            print(f'Detected high error in sqrtm calculation: {delt}')

    return (diff.dot(diff) + np.trace(cov1)
            + np.trace(cov2) - 2 * tr_covmean)


def get_no_cache_embedding_path(audio_file: Union[str, Path], embs: Union[str, Path]) -> Path:
    """
    Get the path to the cached embedding npy file for an audio file.

    :param model: The name of the model
    :param audio_dir: The path to the audio file
    """
    audio_file = Path(audio_file)
    emb_subfolder = audio_file.parent.name.replace("audio","emb")
    emb_path = Path(embs) / emb_subfolder / audio_file.with_suffix(".npy").name
    print("EMB PATH: ", emb_path)
    return emb_path


def no_cache_embedding_files(
        files: Union[list[Path],str, Path],
        embs: Union[list[Path], str, Path],
        ml: ModelLoader,
        workers: int = 8,
        batch_size:int =50,
        **kwargs):
    """
    Get embeddings for all audio files in a directory.

    :param workers: basically 1/batch_size
    :param files: is files
    :param ml_fn: A function that returns a ModelLoader instance.
    """
    if isinstance(files, (str, Path)):
        # Collect all wav files but ignore those that have been converted
        files = [f for f in Path(files).rglob('*.wav') if 'converted' not in f.parts]

    # Filter out files that already have embeddings
    files = [f for f in files if not get_no_cache_embedding_path(f,embs).exists()]
    if len(files) == 0:
        print("All files already have embeddings, skipping.")
        return

    print(f"[Frechet Audio Distance] Loading {len(files)} audio files...")
    multiprocessing.set_start_method('spawn', force=True)

    # Determine the batch size based on the number of workers
    for i in range(0, len(files), batch_size):
        chunk = files[i:i+batch_size]
        print(f"Processing chunk {i // batch_size + 1} / {len(files) // batch_size + 1} with {len(chunk)} files...")

        batches = list(np.array_split(chunk, workers))
            # Cache embeddings in parallel
        with torch.multiprocessing.Pool(workers) as pool:
            pool.map(no_cache_embedding_batch, [(b, ml,embs, kwargs) for b in batches])


def no_cache_embedding_batch(args):
    fs: list[Path]
    ml: ModelLoader
    fs, ml,embs, kwargs = args
    fad = NoCacheFAD(ml,embs=embs, **kwargs)
    for f in fs:
        print(f"Loading {f} using {ml.name}")
        fad.cache_embedding_file(f)


class NoCacheFAD(FrechetAudioDistance):
    def __init__(self, *args, embs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.embs = embs

    def cache_embedding_file(self, audio_dir: Union[str, Path]):
        """
        Compute embedding for an audio file and cache it to a file.
        """
        emb_path = get_no_cache_embedding_path(audio_dir, self.embs)
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

    def score(self, baseline: Union[str, Path], eval: Union[str, Path]):
        print("Loading STATS")
        mu_bg, cov_bg = self.load_stats(baseline)
        print("BASELINE STAT LOADED")
        mu_eval, cov_eval = self.load_stats(eval)
        print("EVAL STAT LOADED")
        return calc_frechet_distance(mu_bg, cov_bg, mu_eval, cov_eval)

    def load_stats(self, path: Union[str, Path]):
        """
        Load embedding statistics from a directory.
        """
        if isinstance(path, str):
            # Check if it's a pre-computed statistic file
            bp = Path(__file__).parent / "stats"
            stats = bp / (path.lower() + ".npz")
            print(stats)
            if stats.exists():
                path = stats

        path = Path(path)

        # Check if path is a file
        if path.is_file():
            # Load it as a npz
            print(f"Loading embedding statistics from {path}...")
            with np.load(path) as data:
                if f'{self.ml.name}.mu' not in data or f'{self.ml.name}.cov' not in data:
                    raise ValueError(f"FAD statistics file {path} doesn't contain data for model {self.ml.name}")
                return data[f'{self.ml.name}.mu'], data[f'{self.ml.name}.cov']

        cache_dir = path / "stats" / self.ml.name
        emb_dir = path / "embeddings" / self.ml.name
        if cache_dir.exists():
            print(f"Embedding statistics is already cached for {path}, loading...")
            mu = np.load(cache_dir / "mu.npy")
            cov = np.load(cache_dir / "cov.npy")
            return mu, cov

        if not path.is_dir():
            print(f"The dataset you want to use ({path}) is not a directory nor a file.")
            exit(1)

        print(f"Loading embedding files from {path}...")

        mu, cov = calculate_embd_statistics_online(list(emb_dir.glob("*.npy")))
        print("> Embeddings statistics calculated.")

        # Save statistics
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "mu.npy", mu)
        np.save(cache_dir / "cov.npy", cov)

        return mu, cov