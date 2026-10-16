"""Download the pinned official ESCI product table with resumable HTTP ranges."""

import argparse
import hashlib
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


SIZE = 1108857465
SHA256 = "25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265"
URL = ("https://media.githubusercontent.com/media/amazon-science/esci-data/main/"
       "shopping_queries_dataset/shopping_queries_dataset_products.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=URL)
    parser.add_argument("--output", type=Path, default=Path("data/raw/esci/products.parquet"))
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    if args.output.exists():
        with args.output.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() == SHA256:
                print("Official product table already verified", flush=True)
                return
        raise ValueError("Existing product table has an unexpected SHA-256")
    parts_dir = args.output.with_suffix(".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    count = 16
    chunk_size = (SIZE + count - 1) // count

    def download(index: int) -> Path:
        start = index * chunk_size
        end = min(SIZE, start + chunk_size) - 1
        part = parts_dir / str(index)
        expected = end - start + 1
        if part.exists() and part.stat().st_size == expected:
            return part
        for attempt in range(3):
            try:
                with requests.get(
                    args.url, params={"segment": index}, headers={"Range": f"bytes={start}-{end}"},
                    stream=True, timeout=(15, 90),
                ) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}":
                        raise ValueError("Download server did not honor the requested HTTP range")
                    with part.open("wb") as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            handle.write(chunk)
                if part.stat().st_size != expected:
                    raise ValueError("Incomplete download segment")
                return part
            except (requests.RequestException, ValueError):
                if attempt == 2:
                    raise
        raise RuntimeError("Download failed")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, index) for index in range(count)]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            print(f"Downloaded {completed}/{count} verified-size segments", flush=True)
    partial = args.output.with_suffix(".part")
    with partial.open("wb") as handle:
        for index in range(count):
            with (parts_dir / str(index)).open("rb") as source:
                shutil.copyfileobj(source, handle, length=1024 * 1024)
    with partial.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != SHA256:
            raise ValueError("Official Git LFS SHA-256 verification failed")
    partial.replace(args.output)
    for index in range(count):
        (parts_dir / str(index)).unlink()
    parts_dir.rmdir()
    print(f"Verified official product table: {SHA256}", flush=True)


if __name__ == "__main__":
    main()
