import argparse
import glob
import multiprocessing
import os
import shutil
from pathlib import Path
import cv2

import tensorflow as tf
import numpy as np

base_path = '/storage/brno12-cerit/home/xlapsa00/datasets/FaceForensics_images'
dataset_path = '/storage/brno12-cerit/home/xlapsa00/datasets/FaceForensics_dataset'
# mask_size = 224  # B0
# mask_size = 384  # V2S
# mask_size = 480  # V2M


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess images from faceforensics')
    parser.add_argument('-m', action='store', dest='mask',
                        help='Mask size', required=True, default=None)

    return parser.parse_args()


def init_worker(size):
    global mask_size
    mask_size = size


def dataset_task(dataset):
    # dataset.map(copy_file)
    global type
    global real
    global mask_size
    if type == 0:
        path = f"{dataset_path}_{mask_size}/train"
    elif type == 1:
        path = f"{dataset_path}_{mask_size}/test"
    else:
        path = f"{dataset_path}_{mask_size}/val"
    if real:
        path = f"{path}/real"
    else:
        path = f"{path}/fake"

    image_path = dataset[0].numpy().decode()
    mask_path = dataset[1].numpy().decode()
    name = os.path.basename(image_path)
    new_image_path = f"{path}/raw/{name}"
    new_mask_path = f"{path}/masks/{name}"

    if os.path.exists(new_image_path):
        cnt = 0
        new_image_path = f"{new_image_path[:-4]}_{cnt}.png"
        new_mask_path = f"{new_mask_path[:-4]}_{cnt}.png"

        while os.path.exists(new_image_path):
            cnt += 1
            new_image_path = f"{new_image_path[:-5]}{cnt}.png"
            new_mask_path = f"{new_mask_path[:-5]}{cnt}.png"
    try:
        # print(f"{image_path}")
        shutil.copyfile(image_path, new_image_path)
        try:
            # print(f"{mask_path}")
            shutil.copyfile(mask_path, new_mask_path)
        except Exception as e:
            os.remove(new_image_path)
            print(f"Unable to copy file mask {new_mask_path}. {e}")
    except Exception as e:
        print(f"Unable to copy file image {name}. {e}")


def prepare_dataset():
    arguments = parse_args()

    global mask_size
    mask_size = int(arguments.mask)

    fakes = glob.glob(f"{base_path}_{mask_size}/fake/*/raw/*.png")
    fake_masks = [f.replace("/raw/", "/masks/") for f in fakes]
    fake_dataset = tf.data.Dataset.from_tensor_slices((fakes, fake_masks))
    reals = glob.glob(f"{base_path}_{mask_size}/real/raw/*.png")
    # Create real mask
    Path(f"{base_path}_{mask_size}/real/masks").mkdir(parents=True, exist_ok=True)
    black_image = 255 * np.zeros((mask_size, mask_size, 3), dtype=np.uint8)
    cv2.imwrite(f"{base_path}_{mask_size}/real/masks/mask.png", black_image)
    real_masks = [f"{base_path}_{mask_size}/real/masks/mask.png"] * len(reals)
    real_dataset = tf.data.Dataset.from_tensor_slices((reals, real_masks))
    # real_dataset = tuple(zip(reals, real_masks))

    Path(f"{dataset_path}_{mask_size}/test/real/raw").mkdir(parents=True, exist_ok=True)
    Path(f"{dataset_path}_{mask_size}/test/real/masks").mkdir(parents=True, exist_ok=True)
    Path(f"{dataset_path}_{mask_size}/test/fake/raw").mkdir(parents=True, exist_ok=True)
    Path(f"{dataset_path}_{mask_size}/test/fake/masks").mkdir(parents=True, exist_ok=True)
    Path(f"{dataset_path}_{mask_size}/train/real/raw").mkdir(parents=True, exist_ok=True)
    Path(f"{dataset_path}_{mask_size}/train/real/masks").mkdir(parents=True, exist_ok=True)
    Path(f"{dataset_path}_{mask_size}/train/fake/raw").mkdir(parents=True, exist_ok=True)
    Path(f"{dataset_path}_{mask_size}/train/fake/masks").mkdir(parents=True, exist_ok=True)
    Path(f"{dataset_path}_{mask_size}/val/real/raw").mkdir(parents=True, exist_ok=True)
    Path(f"{dataset_path}_{mask_size}/val/real/masks").mkdir(parents=True, exist_ok=True)
    Path(f"{dataset_path}_{mask_size}/val/fake/raw").mkdir(parents=True, exist_ok=True)
    Path(f"{dataset_path}_{mask_size}/val/fake/masks").mkdir(parents=True, exist_ok=True)

    # Prepare real part
    dataset_size = len(real_dataset)
    print(f"Real dataset size: {dataset_size}")

    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = int(0.15 * dataset_size)
    print(f"Real dataset train: {train_size} test: {test_size} val: {val_size}")

    dataset = real_dataset.shuffle(dataset_size)
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)

    # Copy to dirs
    global type
    global real
    type = 0
    real = True
    with multiprocessing.Pool(initializer=init_worker, initargs=(mask_size, ),
                              processes=multiprocessing.cpu_count()) as pool:
        _ = pool.map(dataset_task, train_dataset)
    print("Real train done")
    type = 1
    with multiprocessing.Pool(initializer=init_worker, initargs=(mask_size, ),
                              processes=multiprocessing.cpu_count()) as pool:
        _ = pool.map(dataset_task, test_dataset)
    print("Real test done")
    type = 2
    with multiprocessing.Pool(initializer=init_worker, initargs=(mask_size, ),
                              processes=multiprocessing.cpu_count()) as pool:
        _ = pool.map(dataset_task, val_dataset)
    print("Real val done")

    # Prepare fake part
    dataset_size = len(fake_dataset)
    print(f"Fake dataset size: {dataset_size}")

    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = int(0.15 * dataset_size)
    print(f"Fake dataset train: {train_size} test: {test_size} val: {val_size}")

    dataset = fake_dataset.shuffle(dataset_size)
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)

    # Copy to dirs
    type = 0
    real = False
    with multiprocessing.Pool(initializer=init_worker, initargs=(mask_size, ),
                              processes=multiprocessing.cpu_count()) as pool:
        _ = pool.map(dataset_task, train_dataset)
    print("Fake train done")
    type = 1
    with multiprocessing.Pool(initializer=init_worker, initargs=(mask_size, ),
                              processes=multiprocessing.cpu_count()) as pool:
        _ = pool.map(dataset_task, test_dataset)
    print("Fake test done")
    type = 2
    with multiprocessing.Pool(initializer=init_worker, initargs=(mask_size, ),
                              processes=multiprocessing.cpu_count()) as pool:
        _ = pool.map(dataset_task, val_dataset)
    print("Fake val done")


if __name__ == "__main__":
    prepare_dataset()
