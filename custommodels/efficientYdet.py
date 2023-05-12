import tensorflow as tf
from keras import Model
from keras.layers import *
# from keras.metrics import *
from keras.regularizers import l2
from keras.applications import *
from tensorflow import keras
from keras import backend as K
from keras import layers
from keras.models import Sequential
from IPython.display import SVG

import numpy as np

import generators.generators
import models
from processing import checkpoint


def create_bifpn_layer(C, num_channels):
    def _create_bifpn_layer(inputs):
        input_shape = tf.shape(inputs[0])
        H, W = input_shape[1], input_shape[2]
        outputs = []
        for input_tensor in inputs:
            output_tensor = layers.Conv2D(num_channels, 1, padding='same')(input_tensor)
            output_tensor = layers.BatchNormalization()(output_tensor)
            output_tensor = layers.ReLU()(output_tensor)
            resized_output = tf.image.resize(output_tensor, (H, W), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            outputs.append(resized_output)

        output_tensor = layers.Add()(outputs)
        output_tensor = layers.Conv2D(num_channels, 3, padding='same')(output_tensor)
        output_tensor = layers.BatchNormalization()(output_tensor)
        output_tensor = layers.ReLU()(output_tensor)
        return output_tensor

    return _create_bifpn_layer


# EfficientDet models
def create_efficientdetM_binary_classification(num_channels, dropout_rate=0.5):
    backbone = EfficientNetV2M(include_top=False, input_shape=(480, 480, 3), weights='imagenet')
    C3, C4, C5 = backbone.layers[-188].output, backbone.layers[-94].output, backbone.layers[-1].output
    C3 = backbone.get_layer('block3e_expand_activation').output
    C4 = backbone.get_layer('block5n_expand_activation').output
    C5 = backbone.get_layer('block7e_expand_activation').output

    P3 = create_bifpn_layer(C3, num_channels)([C3])
    P4 = create_bifpn_layer(C4, num_channels)([C4, P3])
    P5 = create_bifpn_layer(C5, num_channels)([C5, P4])

    P4 = create_bifpn_layer(C4, num_channels)([C4, P3, P5])
    P5 = create_bifpn_layer(C5, num_channels)([C5, P4])

    P3 = layers.GlobalAveragePooling2D()(P3)
    P4 = layers.GlobalAveragePooling2D()(P4)
    P5 = layers.GlobalAveragePooling2D()(P5)

    pooled_features = layers.Concatenate()([P3, P4, P5])
    dense = layers.Dense(128, activation='relu')(pooled_features)
    dropout = layers.Dropout(dropout_rate)(dense)
    output = layers.Dense(1, activation='sigmoid', name="prediction")(dropout)

    model = Model(inputs=backbone.input, outputs=output)
    return model


def create_efficientdetL_binary_classification_bigger(num_channels, l2_regularization=0.01, dropout_rate=0.5):
    backbone = EfficientNetV2L(include_top=False, input_shape=(480, 480, 3), weights='imagenet')
    C3 = backbone.get_layer('block3g_expand_activation').output
    C4 = backbone.get_layer('block5s_expand_activation').output
    C5 = backbone.get_layer('block7g_expand_activation').output

    P3 = create_bifpn_layer(C3, num_channels)([C3])
    P4 = create_bifpn_layer(C4, num_channels)([C4, P3])
    P5 = create_bifpn_layer(C5, num_channels)([C5, P4])

    P4 = create_bifpn_layer(C4, num_channels)([C4, P3, P5])
    P5 = create_bifpn_layer(C5, num_channels)([C5, P4])

    P3 = layers.GlobalAveragePooling2D()(P3)
    P4 = layers.GlobalAveragePooling2D()(P4)
    P5 = layers.GlobalAveragePooling2D()(P5)

    pooled_features = layers.Concatenate()([P3, P4, P5])
    dense1 = layers.Dense(256, activation='relu', kernel_regularizer=l2(l2_regularization))(pooled_features)
    batch_norm1 = BatchNormalization()(dense1)
    dropout1 = layers.Dropout(dropout_rate)(batch_norm1)

    dense2 = layers.Dense(128, activation='relu', kernel_regularizer=l2(l2_regularization))(dropout1)
    batch_norm2 = BatchNormalization()(dense2)
    dropout2 = layers.Dropout(dropout_rate)(batch_norm2)

    output = layers.Dense(1, activation='sigmoid', name="prediction")(dropout2)

    model = Model(inputs=backbone.input, outputs=output)
    return model


# U-net like connections
def set_unet(efficientdet_model, eff_type):
    # efficientdet_model.summary()
    # Edit efficientDet
    prediction = efficientdet_model.output
    if eff_type == "M":
        # Blocks for V2M
        # block7_output = efficientdet_model.get_layer('block7e_project_bn').output
        # block6_output = efficientdet_model.get_layer('block6r_project_bn').output
        block5_output = efficientdet_model.get_layer('block5n_project_bn').output
        block4_output = efficientdet_model.get_layer('block4g_project_bn').output
        block3_output = efficientdet_model.get_layer('block3e_project_bn').output
        block2_output = efficientdet_model.get_layer('block2e_project_bn').output
        block1_output = efficientdet_model.get_layer('block1c_project_bn').output
        rescaling_output = efficientdet_model.get_layer('rescaling').output

        # x = efficientnet.get_layer('top_activation').output
        # x = block7_output
        # x = upscale_block(512, x, block5_output)
        x = block5_output
        x = upscale_block(256, x, block3_output)
        x = upscale_block(128, x, block2_output)
        x = upscale_block(64, x, block1_output)
        x = upscale_block(32, x, rescaling_output)
        x = Conv2D(1, (1, 1), padding='same')(x)
        reconstruction = Activation('sigmoid', name="reconstruction")(x)

        return Model(inputs=efficientdet_model.input, outputs=[prediction, reconstruction])
    elif eff_type == "L":
        # Blocks for V2L
        # block7_output = efficientdet_model.get_layer('block7g_project_bn').output
        # block6_output = efficientdet_model.get_layer('block6y_project_bn').output
        block5_output = efficientdet_model.get_layer('block5s_project_bn').output
        block4_output = efficientdet_model.get_layer('block4j_project_bn').output
        block3_output = efficientdet_model.get_layer('block3g_project_bn').output
        block2_output = efficientdet_model.get_layer('block2g_project_bn').output
        block1_output = efficientdet_model.get_layer('block1d_project_bn').output
        rescaling_output = efficientdet_model.get_layer('rescaling').output

        # x = efficientnet.get_layer('top_activation').output
        # x = block7_output
        # x = upscale_block(512, x, block5_output)
        x = block5_output
        x = upscale_block(256, x, block3_output)
        x = upscale_block(128, x, block2_output)
        x = upscale_block(64, x, block1_output)
        x = upscale_block(32, x, rescaling_output)
        x = Conv2D(1, (1, 1), padding='same')(x)
        reconstruction = Activation('sigmoid', name="reconstruction")(x)

        return Model(inputs=efficientdet_model.input, outputs=[prediction, reconstruction])

# Losses

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred, smooth=1e-6):
    return 1 - dice_coef(y_true, y_pred, smooth)


# Building blocks

def upscale_block(filters, input_tensor, skip_tensor):
    kernel_size = (3, 3)
    transpose_kernel_size = (2, 2)
    upsample_rate = (2, 2)

    x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate, padding='same')(input_tensor)
    x = Concatenate()([x, skip_tensor])
    x = Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def build_model(trained, eff_type="M", frozen=None, lr=0.0001, dropout_rate=0.5):
    models.models.input_shape = (480, 480, 3)
    # Efficient net has build in preprocessing
    generators.generators.preprocessing_f = None

    # B0, v2B0, S, M, L
    if eff_type == "M":
        efficientdet_model = create_efficientdetM_binary_classification(num_channels=64, dropout_rate=dropout_rate)
    elif eff_type == "L":
        efficientdet_model = create_efficientdetL_binary_classification_bigger(num_channels=64, dropout_rate=dropout_rate)

    if frozen is not None:
        for layer in efficientdet_model.layers:
            layer.trainable = False
            print(f"Layer {layer.name} frozen")
            if layer.name == frozen:
                break

    model = set_unet(efficientdet_model, eff_type)

    if frozen is not None:
        if eff_type == "M":
            models.models.callback_list = checkpoint.checkpoint_callback(f"efficientYdetM-f-{frozen}-lr{lr}-dr{dropout_rate}")
        elif eff_type == "L":
            models.models.callback_list = checkpoint.checkpoint_callback(f"efficientYdetL-f-{frozen}-lr{lr}-dr{dropout_rate}")
    else:
        if eff_type == "M":
            models.models.callback_list = checkpoint.checkpoint_callback(f"efficientYdetM-lr{lr}-dr{dropout_rate}")
        elif eff_type == "L":
            models.models.callback_list = checkpoint.checkpoint_callback(f"efficientYdetL-lr{lr}-dr{dropout_rate}")
    print("done")

    loss_weights = {'prediction': 0.5, 'reconstruction': 0.5}
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss={
            'prediction': 'binary_crossentropy',
            'reconstruction': dice_coef_loss
        },
        loss_weights=loss_weights,
        metrics={
            'prediction': 'accuracy',
            'reconstruction': 'accuracy'
        }
    )
    # model.summary()
    # print(model.get_layer("activation_9").get_config())

    # keras.utils.plot_model(
    #     model,
    #     to_file="model.png",
    #     show_shapes=True,
    #     show_layer_names=True,
    #     rankdir="TB"
    # )

    return model


if __name__ == "__main__":
    build_model(False)
