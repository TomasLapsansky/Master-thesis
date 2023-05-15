"""
File name: densenet121.py
Author: Tomas Lapsansky (xlapsa00@stud.fit.vutbr.cz)
Description: Densenet121 model representation.
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

    generators.generators.preprocessing_f = tensorflow.keras.applications.densenet.preprocess_input

    if trained:
        densenet121 = keras.applications.DenseNet121(
            include_top=True,
            weights='imagenet',
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation='softmax'
        )

        last_layer = densenet121.get_layer('avg_pool').output
        dense = Dense(1, activation='sigmoid', name='predictions')(last_layer)

        new_model = Model(densenet121.input, dense)
        models.models.callback_list = checkpoint.checkpoint_callback("densenet121-trained")
    else:
        densenet121 = keras.applications.DenseNet121(
            include_top=False,
            weights=None,
            input_tensor=None,
            input_shape=(224, 224, 3),
            pooling=max
        )

        new_model = Sequential([
            densenet121,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        models.models.callback_list = checkpoint.checkpoint_callback("densenet121")

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    new_model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    new_model.build((None, 224, 224, 3))

    new_model.summary()

    return new_model
