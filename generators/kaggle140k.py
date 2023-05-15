"""
File name: kaggle140k.py
Author: Tomas Lapsansky (xlapsa00@stud.fit.vutbr.cz)
Description: Processing of Kaggle 140k real and fake faces dataset.
"""

import os

from keras.preprocessing.image import ImageDataGenerator

import generators
import models

base_path = os.getcwd() + '/../datasets/kaggle-140k-real-and-fake-faces/real_vs_fake/real-vs-fake/'


def init():

    image_gen = ImageDataGenerator(rescale=1. / 255., preprocessing_function=generators.generators.preprocessing_f)
    generators.generators.train_flow = image_gen.flow_from_directory(
        base_path + 'train/',
        target_size=models.models.input_shape[:2],
        batch_size=generators.generators.batch_size,
        class_mode='binary'
    )
    print(generators.generators.train_flow.class_indices)

    generators.generators.valid_flow = image_gen.flow_from_directory(
        base_path + 'valid/',
        target_size=models.models.input_shape[:2],
        batch_size=generators.generators.batch_size,
        class_mode='binary'
    )
    print(generators.generators.valid_flow.class_indices)

    generators.generators.test_flow = image_gen.flow_from_directory(
        base_path + 'test/',
        target_size=(224, 224),
        batch_size=1,
        shuffle=False,
        class_mode='binary'
    )
    print(generators.generators.test_flow.class_indices)

    generators.generators.train_steps = 100000 // generators.generators.batch_size
    generators.generators.valid_steps = 20000 // generators.generators.batch_size
    generators.generators.test_steps = 20000 // generators.generators.batch_size

    generators.generators.is_set = True
