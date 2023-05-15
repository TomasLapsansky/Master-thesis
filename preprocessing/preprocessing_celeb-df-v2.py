"""
File name: preprocessing_celeb-df-v2.py
Author: Tomas Lapsansky (xlapsa00@stud.fit.vutbr.cz)
Description: This file is used for preprocessing the Celeb-DF v2 dataset.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2

from mtcnn import MTCNN

base_path = '/storage/brno12-cerit/home/xlapsa00/datasets/Celeb-DF-v2_images_fps_28'
dataset_path = '/storage/brno12-cerit/home/xlapsa00/datasets/Celeb-DF-v2_dataset_fps_28'


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess images from celeb-df')
    parser.add_argument('-m', action='store', dest='mask',
                        help='Mask size', required=True, default=None)

    return parser.parse_args()


def task_video(actor_path, detector):
    # global queue
    global actor_num
    global mask_size
    # Init statistic data
    stat_saved = 0
    stat_unsaved = 0
    stat_bordered = 0
    stat_scaled = 0
    stat_no_face_found = 0
    # Load image
    img = cv2.imread(actor_path)
    # Detect faces in the image
    faces = detector.detect_faces(img)
    # Draw a rectangle around each face
    for face in faces:
        x, y, w, h = face['box']
        x = int(x) + int(w / 2)
        y = int(y) + int(h / 2)
        confidence = face['confidence']
        if confidence > 0.95 and w > 75:
            # Load image one more time for correctness
            img = cv2.imread(actor_path)
            # name = f"{dataset_path}_{mask_size}/real/raw/{actor_num}.png"
            name = actor_path.replace(base_path, f"{dataset_path}_{mask_size}")
            name = f"{name[:-4]}_{actor_num}.png"
            actor_num += 1
            # Calculate scaling
            scaling = 1
            while w > int(mask_size) or h > int(mask_size):
                scaling *= 2
                w = int(w / 2)
                h = int(h / 2)
                x = int(x / 2)
                y = int(y / 2)
                stat_scaled = 1
            # Cropped image
            # Scale if needed
            if stat_scaled:
                scale_percent = 100 / scaling  # percentage
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                dim = (width, height)
                # resize image
                img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            h, w, _ = img.shape
            # Calculate borders
            right_border = abs(min(0, int(w) - (int(x) + int(mask_size / 2))))
            left_border = abs(min(0, int(x) - int(mask_size / 2)))
            top_border = abs(min(0, int(y) - int(mask_size / 2)))
            bottom_border = abs(min(0, int(h) - (int(y) + int(mask_size / 2))))
            # Check
            if right_border > 0 or left_border > 0 or top_border > 0 or bottom_border > 0:
                stat_bordered += 1
            # Cut and create new mask
            new_face = img[
                       (int(y) - int(mask_size / 2) + top_border):(int(y) + int(mask_size / 2) - bottom_border),
                       (int(x) - int(mask_size / 2) + left_border):(int(x) + int(mask_size / 2) - right_border)]
            # Fill space with black space
            new_face_image = cv2.copyMakeBorder(new_face, top_border, bottom_border, left_border, right_border,
                                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
            Path(os.path.dirname(name)).mkdir(parents=True, exist_ok=True)
            try:
                if cv2.imwrite(name, new_face_image):
                    # print(name + " real saved")
                    stat_saved += 1
                else:
                    print(name + " real not saved!", file=sys.stderr)
                    stat_unsaved += 1
            except Exception as e:
                print(f"{actor_path} failed due to error: {e}", file=sys.stderr)
    if len(faces) == 0:
        stat_no_face_found = 1
    return tuple([stat_saved, stat_unsaved, stat_bordered, stat_scaled, stat_no_face_found])


def cut_faces():
    arguments = parse_args()

    global mask_size
    mask_size = int(arguments.mask)

    real_paths = []
    for path, subdirs, files in os.walk(base_path):
        real_paths += [os.path.join(path, name) for name in files if
                       name.endswith('.png') != -1]
    print(f"Done loading video paths {len(real_paths)}")
    # Processing videos
    detector = MTCNN()
    global actor_num
    actor_num = 0
    cnt = len(real_paths)
    stats = (0, 0, 0, 0, 0)
    for path in real_paths:
        val = task_video(path, detector)
        if len(val) != 0:
            stats = tuple(sum(a) for a in zip(stats, val))
        cnt -= 1
        if cnt % 100 == 0:
            print(f"{cnt} to be done")
    print(f'Done with second step: {stats}')
    print('Done', flush=True)


if __name__ == "__main__":
    cut_faces()
