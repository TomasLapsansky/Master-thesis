from keras import Model
from keras.layers import Dense
from tensorflow import keras
from keras.models import Sequential
import tensorflow

import generators.generators
import models
from processing import checkpoint


def build_model(trained):
    models.models.input_shape = (224, 224, 3)

    generators.generators.preprocessing_f = tensorflow.keras.applications.vgg19.preprocess_input

    if trained:
        vgg19 = keras.applications.VGG19(
            include_top=True,
            weights='imagenet',
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation='softmax'
        )

        # customize output
        last_layer = vgg19.get_layer('fc2').output
        dense = Dense(1, activation='sigmoid', name='predictions')(last_layer)

        new_model = Model(vgg19.input, dense)
        models.models.callback_list = checkpoint.checkpoint_callback("vgg19-trained")
    else:
        vgg19 = keras.applications.VGG19(
            include_top=False,
            weights=None,
            input_tensor=None,
            input_shape=(224, 224, 3),
            pooling=max
        )

        new_model = Sequential([
            vgg19,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        models.models.callback_list = checkpoint.checkpoint_callback("vgg19")

    new_model.summary()

    new_model.compile(
        optimizer="Adam",
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    new_model.build((None, 224, 224, 3))

    new_model.summary()

    return new_model
