from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import requests


DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_FILES = ("u.data", "u.item", "README")
LIKE_THRESHOLD = 4


@dataclass(frozen=True)
class MovieLens:
    ratings: np.ndarray
    titles: dict[int, str]
    shape: tuple[int, int]

    def rating_matrix(self, ratings: np.ndarray) -> np.ndarray:
        matrix = np.zeros(self.shape, dtype=np.uint8)
        users, movies, stars, _ = ratings.T
        matrix[users - 1, movies - 1] = stars
        return matrix


def download_dataset(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with requests.get(DATA_URL, timeout=(10, 60)) as response:
        response.raise_for_status()
        archive_bytes = response.content

    # Read only the files used by this project, plus the data provider's README.
    with ZipFile(BytesIO(archive_bytes)) as archive:
        with TemporaryDirectory(dir=destination) as temporary:
            staging = Path(temporary)
            for name in DATA_FILES:
                (staging / name).write_bytes(archive.read(f"ml-100k/{name}"))
            for name in DATA_FILES:
                (staging / name).replace(destination / name)


def load_dataset(directory: Path) -> MovieLens:
    if not all((directory / name).is_file() for name in DATA_FILES[:2]):
        print("正在从 GroupLens 下载 MovieLens 100K…", flush=True)
        download_dataset(directory)

    ratings = np.loadtxt(directory / "u.data", dtype=np.int64, delimiter="\t", ndmin=2)
    titles = {}
    for line in (directory / "u.item").read_text(encoding="latin-1").splitlines():
        movie_id, title, _ = line.split("|", maxsplit=2)
        titles[int(movie_id)] = title

    ratings = ratings[np.argsort(ratings[:, 3], kind="stable")]
    return MovieLens(ratings, titles, (int(ratings[:, 0].max()), max(titles)))


def split_by_time(ratings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = ratings[:, 3]
    # Put ratings with the same timestamp in the same window.
    cutoffs = timestamps[[len(ratings) * 70 // 100, len(ratings) * 85 // 100]]
    boundaries = timestamps.searchsorted(cutoffs, side="left")
    train, validation, test = np.split(ratings, boundaries)
    return train, validation, test
