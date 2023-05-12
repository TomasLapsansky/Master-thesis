import tensorflow
from keras import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import *
from tensorflow import keras
from keras.models import Sequential

import generators.generators
import models
from processing import checkpoint


def build_model(trained, eff_type="M", frozen=None, lr=0.0001):
    generators.generators.preprocessing_f = None

    if eff_type == "B0":
        models.models.input_shape = (224, 224, 3)
        efficientnet_model = EfficientNetV2B0(
            include_top=False,
            weights='imagenet'
        )
    elif eff_type == "M":
        models.models.input_shape = (480, 480, 3)
        efficientnet_model = EfficientNetV2M(
            include_top=False,
            weights='imagenet'
        )
    else:
        models.models.input_shape = (480, 480, 3)
        efficientnet_model = EfficientNetV2L(
            include_top=False,
            weights='imagenet'
        )

    # if v2:
    #     efficientnet_model = EfficientNetV2B0(
    #         include_top=False,
    #         weights='imagenet'
    #     )
    # else:
    #     efficientnet_model = EfficientNetB0(
    #         include_top=False,
    #         weights='imagenet'
    #     )
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
        if eff_type == "B0":
            models.models.callback_list = checkpoint.checkpoint_callback(f"efficientnetB0-f-{frozen}")
        elif eff_type == "M":
            models.models.callback_list = checkpoint.checkpoint_callback(f"efficientnetM-f-{frozen}-lr{lr}")
        elif eff_type == "L":
            models.models.callback_list = checkpoint.checkpoint_callback(f"efficientnetL-f-{frozen}-lr{lr}")
    else:
        if eff_type == "B0":
            models.models.callback_list = checkpoint.checkpoint_callback(f"efficientnetB0")
        elif eff_type == "M":
            models.models.callback_list = checkpoint.checkpoint_callback(f"efficientnetM-lr{lr}")
        elif eff_type == "L":
            models.models.callback_list = checkpoint.checkpoint_callback(f"efficientnetL-lr{lr}")
        # models.models.callback_list = checkpoint.checkpoint_callback(f"efficientnet")

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    # optimizer = keras.optimizers.Adam()
    new_model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # new_model.summary()
    return new_model
