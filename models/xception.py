import tensorflow
from keras import Model
from keras.layers import Dense
from tensorflow import keras
from keras.models import Sequential

import generators.generators
import models
from processing import checkpoint


def build_model(trained):
    models.models.input_shape = (224, 224, 3)  # tmp

    generators.generators.preprocessing_f = tensorflow.keras.applications.xception.preprocess_input

    if trained:
        xception = keras.applications.xception.Xception(
            include_top=True,
            weights='imagenet',
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation='softmax'
        )

        # customize output
        last_layer = xception.get_layer('avg_pool').output
        dense = Dense(1, activation='sigmoid', name='predictions')(last_layer)

        new_model = Model(xception.input, dense)
        models.models.callback_list = checkpoint.checkpoint_callback("xception-trained")
    else:
        xception = keras.applications.xception.Xception(
            include_top=False,
            weights=None,
            input_tensor=None,
            input_shape=(224, 224, 3),
            pooling=max,
        )

        new_model = Sequential([
            xception,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        models.models.callback_list = checkpoint.checkpoint_callback("xception")

    new_model.compile(
        optimizer="Adam",
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    new_model.build((None, 224, 224, 3))

    new_model.summary()

    return new_model
