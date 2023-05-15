"""
File name: preprocessing_faceforensics_masks.py
Author: Tomas Lapsansky (xlapsa00@stud.fit.vutbr.cz)
Description: This file is used for preprocessing the FaceForensics dataset.
"""

import argparse
import multiprocessing
import os
import sys
from multiprocessing import SimpleQueue
from pathlib import Path

import numpy as np
import cv2

from mtcnn import MTCNN

base_path = '/storage/brno12-cerit/home/xlapsa00/datasets/FaceForensics_new_images'
dataset_path = '/storage/brno12-cerit/home/xlapsa00/datasets/FaceForensics_new_dataset'


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess images from faceforensics')
    parser.add_argument('-m', action='store', dest='mask',
                        help='Mask size', required=True, default=None)

    return parser.parse_args()


def init_worker(shared_queue, size):
    global queue
    global mask_size
    queue = shared_queue
    mask_size = size


def task_masks(mask_path):
    global queue
    global mask_size
    # Init statistic data
    stat_mask_saved = 0
    stat_mask_unsaved = 0
    stat_raw_saved = 0
    stat_raw_unsaved = 0
    stat_raw_non_exist = 0
    stat_bordered = 0
    stat_scaled = 0
    stat_real_saved = 0
    stat_real_unsaved = 0
    stat_real_non_exist = 0
    # Load image
    img = cv2.imread(mask_path)

    # Convert to grayscale and threshold
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 1, 255, 0)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        # Wrong images without masks
        queue.put(tuple([]))
        return
    if len(contours) > 1:
        # Pick main contour
        main_radius = 0
        main_center = (int(0), int(0))

        for cnt in contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            if radius > 0:
                if radius > main_radius:
                    main_radius = radius
                    main_center = center
        radius = main_radius
        center = main_center
    else:
        (x, y), radius = cv2.minEnclosingCircle(contours[0])
        center = (int(x), int(y))
        radius = int(radius)
    # Calculate scaling
    scaling = 1
    while radius > int(mask_size / 2):
        scaling *= 2
        radius = int(radius / 2)
        center = (int(center[0] / 2), int(center[1] / 2))
        stat_scaled = 1
    # Cropped mask
    # Fill white mask
    img = cv2.drawContours(img, contours, -1, color=(255, 255, 255), thickness=cv2.FILLED)
    # Scale if needed
    if stat_scaled:
        scale_percent = 100/scaling  # percentage
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    h, w, _ = img.shape
    # Calculate borders
    right_border = abs(min(0, w - (center[0] + int(mask_size / 2))))
    left_border = abs(min(0, center[0] - int(mask_size / 2)))
    top_border = abs(min(0, center[1] - int(mask_size / 2)))
    bottom_border = abs(min(0, h - (center[1] + int(mask_size / 2))))
    # Check
    if right_border > 0 or left_border > 0 or top_border > 0 or bottom_border > 0:
        stat_bordered += 1
    # Cut and create new mask
    new_mask = img[(center[1] - int(mask_size / 2) + top_border):(center[1] + int(mask_size / 2) - bottom_border),
               (center[0] - int(mask_size / 2) + left_border):(center[0] + int(mask_size / 2) - right_border)]
    # Fill space with black space
    new_mask_image = cv2.copyMakeBorder(new_mask, top_border, bottom_border, left_border, right_border,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    kernel = np.ones((5, 5), np.uint8)
    new_mask_image = cv2.morphologyEx(new_mask_image, cv2.MORPH_CLOSE, kernel)
    new_mask_image = cv2.morphologyEx(new_mask_image, cv2.MORPH_OPEN, kernel)

    # Saving mask
    new_mask_name = mask_path.replace(base_path, f"{dataset_path}_{mask_size}")
    Path(os.path.dirname(new_mask_name)).mkdir(parents=True, exist_ok=True)
    try:
        if cv2.imwrite(new_mask_name, new_mask_image):
            # print(new_mask_name + " mask saved")
            stat_mask_saved += 1
        else:
            print(new_mask_name + " mask not saved!", file=sys.stderr)
            stat_mask_unsaved += 1
    except Exception as e:
        print(f"{new_mask_name} failed due to error: {e}", file=sys.stderr)

    # Load raw image if exists
    raw_path = mask_path.replace("/masks/", "/raw/")
    if os.path.exists(raw_path):
        img_raw = cv2.imread(raw_path)
        # Scale if needed
        if stat_scaled:
            scale_percent = 100/scaling  # percentage
            width = int(img_raw.shape[1] * scale_percent / 100)
            height = int(img_raw.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            img_raw = cv2.resize(img_raw, dim, interpolation=cv2.INTER_AREA)
        # Cut and create new raw image
        new_raw = img_raw[(center[1] - int(mask_size / 2) + top_border):(center[1] + int(mask_size / 2) - bottom_border),
                  (center[0] - int(mask_size / 2) + left_border):(center[0] + int(mask_size / 2) - right_border)]
        # Fill space with black space
        new_raw_image = cv2.copyMakeBorder(new_raw, top_border, bottom_border, left_border, right_border,
                                           cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # Saving raw
        new_raw_name = raw_path.replace(base_path, f"{dataset_path}_{mask_size}")
        # new_raw_name = new_raw_name.replace("/manipulated_sequences/", "/fake/")
        # new_raw_name = new_raw_name.replace("/videos/", "/")
        Path(os.path.dirname(new_raw_name)).mkdir(parents=True, exist_ok=True)
        try:
            if cv2.imwrite(new_raw_name, new_raw_image):
                # print(new_raw_name + " raw saved")
                stat_raw_saved += 1
            else:
                print(new_raw_name + " raw not saved!", file=sys.stderr)
                stat_raw_unsaved += 1
        except Exception as e:
            print(f"{new_mask_name} failed due to error: {e}", file=sys.stderr)
    else:
        stat_raw_non_exist += 1

    queue.put(tuple([stat_mask_saved, stat_mask_unsaved, stat_raw_saved, stat_raw_unsaved, stat_raw_non_exist,
                     stat_bordered, stat_scaled, stat_real_saved, stat_real_unsaved, stat_real_non_exist]))


def task_actors(actor_path, detector):
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
            mask_name = name.replace("/raw/", "/masks/")
            Path(os.path.dirname(name)).mkdir(parents=True, exist_ok=True)
            Path(os.path.dirname(mask_name)).mkdir(parents=True, exist_ok=True)
            black_image = 255 * np.zeros((mask_size, mask_size, 3), dtype=np.uint8)
            try:
                if cv2.imwrite(mask_name, black_image):
                    pass
                else:
                    print(name + " real mask not saved!", file=sys.stderr)
            except Exception as e:
                print(f"{actor_path} failed due to error: {e}", file=sys.stderr)
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


def cut_masks():
    arguments = parse_args()

    global mask_size
    mask_size = int(arguments.mask)
    # return
    shared_queue = SimpleQueue()
    mask_paths = []
    for path, subdirs, files in os.walk(base_path):
        # print(path)
        mask_paths += [os.path.join(path, name) for name in files if
                       name.endswith('.png') and path.find("/masks") != -1]
    print(f"Done loading mask paths {len(mask_paths)}")
    # Processing mask paths and their raws, reals from actors
    cnt = len(mask_paths)
    with multiprocessing.Pool(initializer=init_worker, initargs=(shared_queue, mask_size, ),
                              processes=multiprocessing.cpu_count()) as pool:
        _ = pool.map_async(task_masks, mask_paths)
        stats = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        for _ in mask_paths:
            val = shared_queue.get()
            if len(val) != 0:
                stats = tuple(sum(a) for a in zip(stats, val))
            cnt -= 1
            if cnt % 1000 == 0:
                print(f"{cnt} to be done")
    print(f'Done with first step: {stats}')
    real_paths = []
    for path, subdirs, files in os.walk(base_path):
        real_paths += [os.path.join(path, name) for name in files if
                       name.endswith('.png') and path.find("/real") != -1]
    print(f"Done loading real paths {len(real_paths)}")
    # Processing reals from YouTube
    detector = MTCNN()
    global actor_num
    actor_num = 0
    cnt = len(real_paths)
    stats = (0, 0, 0, 0, 0)
    for path in real_paths:
        val = task_actors(path, detector)
        if len(val) != 0:
            stats = tuple(sum(a) for a in zip(stats, val))
        cnt -= 1
        if cnt % 100 == 0:
            print(f"{cnt} to be done")
    print(f'Done with second step: {stats}')
    print('Done', flush=True)


if __name__ == "__main__":
    cut_masks()
