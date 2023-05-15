"""
File name: jpeg_compression.py
Author: Tomas Lapsansky (xlapsa00@stud.fit.vutbr.cz)
Description: This file is used for jpeg compression of folder, currently FaceForensics dataset
"""

from pathlib import Path
from PIL import Image
import argparse
import multiprocessing


def compress_image(params):
    input_path, output_path, quality = params
    try:
        img = Image.open(input_path)
        img.save(output_path, "JPEG", quality=quality)
        print(f"Compressed {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error compressing {input_path}: {e}")


def process_directory(source_dir, target_dir, quality=80, exts=None):
    if exts is None:
        exts = [".jpg", ".jpeg", ".png"]
    source = Path(source_dir)
    target = Path(target_dir)
    tasks = []

    for input_path in source.glob("**/*"):
        if input_path.is_file() and input_path.suffix.lower() in exts:
            relative_path = input_path.relative_to(source)
            output_path = target / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((input_path, output_path, quality))

    with multiprocessing.Pool() as pool:
        pool.map(compress_image, tasks)


def main():
    parser = argparse.ArgumentParser(description="Compress images in a directory and its subdirectories.")
    parser.add_argument("-q", "--quality", type=int, default=80, help="Compression quality (0-100)")

    args = parser.parse_args()

    source_directory = "/storage/brno12-cerit/home/xlapsa00/datasets/FaceForensics_new_dataset_480"
    target_directory = f"/storage/brno12-cerit/home/xlapsa00/datasets/FaceForensics_new_dataset_480_jpeg_{args.quality}"
    compression_quality = args.quality

    process_directory(source_directory, target_directory, compression_quality)


if __name__ == "__main__":
    main()

