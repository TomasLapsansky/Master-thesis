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

    generators.generators.preprocessing_f = tensorflow.keras.applications.resnet_v2.preprocess_input

    if trained:
        resnet50v2 = keras.applications.resnet_v2.ResNet50V2(
            include_top=True,
            weights='imagenet',
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation='softmax'
        )

        last_layer = resnet50v2.get_layer('avg_pool').output
        dense = Dense(1, activation='sigmoid', name='predictions')(last_layer)

        new_model = Model(resnet50v2.input, dense)
        models.models.callback_list = checkpoint.checkpoint_callback("resnet50v2-trained")
    else:
        resnet50v2 = keras.applications.resnet_v2.ResNet50V2(
            include_top=False,
            weights=None,
            input_tensor=None,
            input_shape=(224, 224, 3),
            pooling=max
        )

        new_model = Sequential([
            resnet50v2,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        models.models.callback_list = checkpoint.checkpoint_callback("resnet50v2")

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    new_model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    new_model.build((None, 224, 224, 3))

    new_model.summary()

    return new_model
