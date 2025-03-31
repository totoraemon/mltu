"""
Data utilities for downloading, unzipping, and processing handwriting datasets.
"""

import os
import concurrent.futures
from tqdm import tqdm
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
from typing import Optional, Tuple, List, Set

def download_and_unzip(url: str, extract_to: str = "Datasets", chunk_size: int = 1024 * 1024) -> None:
    """
    Downloads a ZIP file from a URL and extracts it to the specified location.

    Args:
        url (str): The URL of the ZIP file to download.
        extract_to (str): The directory to extract files to. Defaults to "Datasets".
        chunk_size (int): The size (in bytes) of each chunk to read from the URL.
            Defaults to 1024 * 1024.

    Raises:
        ConnectionError: If the URL cannot be opened.
        OSError: If the ZIP file cannot be opened or extracted.
    """
    try:
        http_response = urlopen(url)
    except Exception as e:
        raise ConnectionError(f"Failed to open URL: {url}") from e

    if not hasattr(http_response, "length") or http_response.length is None:
        raise OSError("Could not determine file size for download.")

    iterations = (http_response.length // chunk_size) + 1
    chunks = []
    for _ in tqdm(range(iterations), desc="Downloading"):
        chunks.append(http_response.read(chunk_size))

    data = b"".join(chunks)
    try:
        zipfile = ZipFile(BytesIO(data))
        zipfile.extractall(path=extract_to)
    except Exception as e:
        raise OSError("Failed to extract ZIP file.") from e


def process_line(line: str, dataset_path: str) -> Optional[Tuple[str, str]]:
    """
    Processes a single line from the dataset metadata file to extract
    the image path and label.

    Args:
        line (str): A single line containing image metadata.
        dataset_path (str): The base path of the dataset.

    Returns:
        Optional[Tuple[str, str]]:
            A tuple of (relative_path, label) if valid, otherwise None.
    """
    if line.startswith("#"):
        return None

    line_split = line.split(" ")
    if len(line_split) < 2 or line_split[1] == "err":
        return None

    folder1 = line_split[0][:3]
    folder2 = "-".join(line_split[0].split("-")[:2])
    file_name = line_split[0] + ".png"
    label = line_split[-1].rstrip("\n")

    rel_path = os.path.join(dataset_path, "words", folder1, folder2, file_name)
    if not os.path.exists(rel_path):
        return None

    return rel_path, label


def load_dataset(dataset_path: str) -> Tuple[List[List[str]], Set[str], int]:
    """
    Loads the dataset by reading 'words.txt' in the specified folder,
    processing each line, and collating results.

    Args:
        dataset_path (str): The path to the dataset folder containing 'words.txt'.

    Returns:
        Tuple[List[List[str]], Set[str], int]:
            A tuple containing:
            - dataset (List[List[str]]): A list of [rel_path, label] entries.
            - vocab (Set[str]): A set of unique characters found in labels.
            - max_len (int): The longest label length.
    """
    dataset: List[List[str]] = []
    vocab: Set[str] = set()
    max_len: int = 0

    words_file = os.path.join(dataset_path, "words.txt")
    if not os.path.exists(words_file):
        raise FileNotFoundError(f"Could not find 'words.txt' at {words_file}")

    with open(words_file, "r", encoding="utf-8") as file_obj:
        lines = file_obj.readlines()
        # Reduced dataset size for demonstration; remove slicing as needed.
        lines = lines[:1000]

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(
            tqdm(
                executor.map(
                    lambda ln: process_line(ln, dataset_path), lines
                ),
                total=len(lines),
                desc="Processing lines"
            )
        )

    for r in results:
        if r is None:
            continue
        rel_path, label = r
        dataset.append([rel_path, label])
        vocab.update(list(label))
        max_len = max(max_len, len(label))

    return dataset, vocab, max_len