import os
import sys
import cv2
import multiprocessing
from pathlib import Path

# base_path = '/storage/brno12-cerit/home/xlapsa00/datasets/DFDC'
# images_dir = '/storage/brno12-cerit/home/xlapsa00/datasets/DFDC_images'
# base_path = '/storage/brno12-cerit/home/xlapsa00/datasets/FaceForensics'
# images_dir = '/storage/brno12-cerit/home/xlapsa00/datasets/FaceForensics_images'
# base_path = '/storage/brno12-cerit/home/xlapsa00/datasets/FaceForensics_new'
# images_dir = '/storage/brno12-cerit/home/xlapsa00/datasets/FaceForensics_new_images'
base_path = '/storage/brno12-cerit/home/xlapsa00/datasets/Celeb-DF'
images_dir = '/storage/brno12-cerit/home/xlapsa00/datasets/Celeb-DF_images_fps_28'
image_fps = 28


def task(video_path):
    filename, file_extension = os.path.splitext(os.path.relpath(video_path, base_path))
    image_path = os.path.join(images_dir, filename)
    extension = ".png"
    try:
        cap = cv2.VideoCapture(video_path)
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        hop = round(fps / image_fps)
        curr_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if curr_frame % hop == 0:
                name = image_path + "_" + str(curr_frame) + extension
                Path(os.path.dirname(name)).mkdir(parents=True, exist_ok=True)

                if cv2.imwrite(name, frame):
                    print(name + " saved")
                else:
                    print(name + " not saved!", file=sys.stderr)
            curr_frame += 1
    except Exception as e:
        print(f"{video_path} failed due to error: {e}", file=sys.stderr)
    finally:
        cap.release()


def convert_to_image():
    video_paths = []
    for path, subdirs, files in os.walk(base_path):
        video_paths += [os.path.join(path, name) for name in files if name.endswith('.mp4')]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(task, video_paths)

    print('Done', flush=True)


if __name__ == "__main__":
    convert_to_image()
