import tensorflow
from keras import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import *
from tensorflow import keras
from keras.models import Sequential

import generators.generators
import models
from processing import checkpoint


def build_model(trained, frozen=None, lr=0.0001):
    models.models.input_shape = (480, 480, 3)

    generators.generators.preprocessing_f = None

    efficientnet_model = EfficientNetV2M(
        include_top=False,
        weights='imagenet'
    )
    if frozen is not None:
        for layer in efficientnet_model.layers:
            layer.trainable = False
            print(f"Layer {layer.name} frozen")
            if layer.name == frozen:
                break

    x = efficientnet_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    prediction = Dense(1, activation='sigmoid', name='prediction')(x)

    new_model = Model(efficientnet_model.input, prediction)
    if frozen is not None:
        models.models.callback_list = checkpoint.checkpoint_callback(f"efficientnetM-f-{frozen}-lr{lr}")
    else:
        models.models.callback_list = checkpoint.checkpoint_callback(f"efficientnetM-lr{lr}")

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    new_model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    new_model.summary()

    return new_model
