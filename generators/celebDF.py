import glob

import numpy as np
import tensorflow as tf
import os
import re

import generators.generators

base_path = '/storage/brno12-cerit/home/xlapsa00/datasets/Celeb-DF_dataset_480'
base_path_100 = '/storage/brno12-cerit/home/xlapsa00/datasets/Celeb-DF_dataset_480_jpeg_100'
base_path_80 = '/storage/brno12-cerit/home/xlapsa00/datasets/Celeb-DF_dataset_480_jpeg_80'
base_path_60 = '/storage/brno12-cerit/home/xlapsa00/datasets/Celeb-DF_dataset_480_jpeg_60'
base_path_40 = '/storage/brno12-cerit/home/xlapsa00/datasets/Celeb-DF_dataset_480_jpeg_40'
input_file_path = '/storage/brno12-cerit/home/xlapsa00/datasets/Celeb-DF/List_of_testing_videos.txt'


def process_line(line):
    global compression
    pattern = r'(\d) (.+)\.mp4'
    matches = re.findall(pattern, line)

    dataset_path = base_path

    if compression == 100:
        dataset_path = base_path_100
    elif compression == 80:
        dataset_path = base_path_80
    elif compression == 60:
        dataset_path = base_path_60
    elif compression == 40:
        dataset_path = base_path_40


    images, labels = [], []
    for match in matches:
        label, video_path_prefix = match
        label = tf.strings.to_number(label, out_type=tf.float32)

        # Find all matching image files using glob
        image_paths = glob.glob(f"{dataset_path}/{video_path_prefix}_*_*[0-9].png")
        for image_path in image_paths:
            img = tf.io.read_file(image_path)
            img = tf.image.decode_png(img)
            img = tf.image.convert_image_dtype(img, tf.float32)

            images.append(img)
            labels.append(label)

    return images, labels


def dataset_size(txt_file_path):
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()

    image_count = 0
    for line in lines:
        images, labels = process_line(line.strip())
        image_count += len(images)

    return image_count


def process_line_generator(txt_file_path):
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        images, labels = process_line(line.strip())
        if not images:
            return None
        else:
            yield images, labels


def process_labels(txt_file_path):
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        images, labels = process_line(line.strip())
        if not images:
            return None
        else:
            yield labels


def init(shape=480, compress=None):
    global compression
    compression = compress

    output_signature = (
        tf.TensorSpec(shape=(None, 480, 480, 3), dtype=tf.float32),
        # tf.TensorSpec(shape=(None,), dtype=tf.string),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
    # Get the dataset size
    dataset_len = dataset_size(input_file_path)
    print(f"Dataset size: {dataset_len}")

    dataset = tf.data.Dataset.from_generator(
        lambda: process_line_generator(input_file_path),
        output_signature=output_signature
    )

    dataset = dataset.unbatch()
    dataset = dataset.batch(generators.generators.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    generators.generators.test_flow = dataset
    true_labels = []

    for _, batch_labels in dataset:
        # Convert the batch labels to a NumPy array if necessary
        if isinstance(batch_labels, tf.Tensor):
            batch_labels = batch_labels.numpy()

        batch_true_labels = np.round(batch_labels).astype(int)

        true_labels.extend(batch_true_labels)

    generators.generators.test_labels = np.array(true_labels)

    generators.generators.test_steps = dataset_len / generators.generators.batch_size

    generators.generators.is_set = True

    # Iterate over the dataset and print the elements
    # for batch_idx, (images, labels) in enumerate(dataset):
    #     print(f"Batch {batch_idx + 1}:")
    #     for idx, (image, label) in enumerate(zip(images.numpy(), labels.numpy())):
    #         print(f"  Image {idx + 1}: shape={image}, label={label}")
    print(f"steps: {generators.generators.test_steps}")
