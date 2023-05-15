"""
File name: resnet50.py
Author: Tomas Lapsansky (xlapsa00@stud.fit.vutbr.cz)
Description: Resnet50 model representation.
"""

import tensorflow
from keras import Model
from keras.layers import Dense
from tensorflow import keras
from keras.models import Sequential

import generators.generators
import models
from processing import checkpoint


def build_model(trained, lr=0.0001):
    models.models.input_shape = (224, 224, 3)

    generators.generators.preprocessing_f = tensorflow.keras.applications.resnet.preprocess_input

    if trained:
        resnet50 = keras.applications.resnet50.ResNet50(
            include_top=True,
            weights='imagenet',
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000
        )

        last_layer = resnet50.get_layer('avg_pool').output
        dense = Dense(1, activation='sigmoid', name='predictions')(last_layer)

        new_model = Model(resnet50.input, dense)
        models.models.callback_list = checkpoint.checkpoint_callback("resnet50-trained")
    else:
        resnet = keras.applications.resnet50.ResNet50(
            include_top=False,
            weights=None,
            input_tensor=None,
            input_shape=(224, 224, 3),
            pooling=max
        )

        new_model = Sequential([
            resnet,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        models.models.callback_list = checkpoint.checkpoint_callback("resnet50")

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    new_model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    new_model.build((None, 224, 224, 3))

    new_model.summary()

    return new_model
