"""
File name: checkpoint.py
Author: Tomas Lapsansky (xlapsa00@stud.fit.vutbr.cz)
Description: This file is used for handling checkpoints in the project.
"""

import os

from keras.callbacks import ModelCheckpoint

import generators


def checkpoint_callback(name, multiple=False):
    if multiple:
        # checkpoint_path = os.getcwd() + "/../checkpoints-ynet"
        checkpoint_path = os.getcwd() + "/../checkpoints"
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        filepath = checkpoint_path + "/" + name + "-COMBINED-" + generators.generators.dataset_name + "-{epoch:02d}-VAL-loss_{" \
                                                                                             "val_loss}-p_loss_{" \
                                                                                             "val_prediction_loss:.2f" \
                                                                                             "}-r_loss_{" \
                                                                                             "val_reconstruction_loss" \
                                                                                             ":.2f}-p_acc_{" \
                                                                                             "val_prediction_accuracy" \
                                                                                             ":.2f}-r_acc_{" \
                                                                                             "val_reconstruction_accuracy:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_reconstruction_accuracy', verbose=1, save_best_only=False, mode='max')
    else:
        checkpoint_path = os.getcwd() + "/../checkpoints"
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        filepath = checkpoint_path + "/" + name + "-" + generators.generators.dataset_name + "-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')
    callbacks_list = [checkpoint]
    return callbacks_list
