"""
File name: efficientYnet.py
Author: Tomas Lapsansky (xlapsa00@stud.fit.vutbr.cz)
Description: Our custom EfficientYnet model representation.
"""

from keras import Model
from keras.layers import *
from keras.applications import *
from tensorflow import keras
from keras import backend as K

import generators.generators
import models
from processing import checkpoint


# EfficientNet models

def get_efficientnet(trained, eff_type, frozen, lr):
    if trained:
        if eff_type == "S":
            print(F"Setting efficientYnetS-pretrained")
            models.models.input_shape = (384, 384, 3)
            if frozen is not None:
                models.models.callback_list = checkpoint.checkpoint_callback(f"efficientYnetS-pretrained-f-{frozen}-lr{lr}", True)
            else:
                models.models.callback_list = checkpoint.checkpoint_callback("efficientYnetS-pretrained", True)
            return EfficientNetV2S(
                include_top=False,
                weights="imagenet",
                input_shape=models.models.input_shape
            )
        elif eff_type == "M":
            print(F"Setting efficientYnetM-pretrained")
            models.models.input_shape = (480, 480, 3)
            if frozen is not None:
                models.models.callback_list = checkpoint.checkpoint_callback(f"efficientYnetM-pretrained-f-{frozen}-lr{lr}", True)
            else:
                models.models.callback_list = checkpoint.checkpoint_callback("efficientYnetM-pretrained", True)
            return EfficientNetV2M(
                include_top=False,
                weights="imagenet",
                input_shape=models.models.input_shape
            )
        elif eff_type == "L":
            print(F"Setting efficientYnetL-pretrained")
            models.models.input_shape = (480, 480, 3)
            if frozen is not None:
                models.models.callback_list = checkpoint.checkpoint_callback(f"efficientYnetL-pretrained-f-{frozen}-lr{lr}", True)
            else:
                models.models.callback_list = checkpoint.checkpoint_callback("efficientYnetL-pretrained", True)
            return EfficientNetV2L(
                include_top=False,
                weights="imagenet",
                input_shape=models.models.input_shape
            )
        elif eff_type == "B0":
            print(F"Setting efficientYnetB0-pretrained")
            models.models.input_shape = (224, 224, 3)
            if frozen is not None:
                models.models.callback_list = checkpoint.checkpoint_callback(f"efficientYnetB0-pretrained-f-{frozen}-lr{lr}", True)
            else:
                models.models.callback_list = checkpoint.checkpoint_callback("efficientYnetB0-pretrained", True)
            return EfficientNetB0(
                include_top=False,
                weights="imagenet",
                input_shape=models.models.input_shape
            )
        elif eff_type == "V2B0":
            print(F"Setting efficientYnetV2B0-pretrained")
            models.models.input_shape = (224, 224, 3)
            if frozen is not None:
                models.models.callback_list = checkpoint.checkpoint_callback(f"efficientYnetV2B0-pretrained-f-{frozen}-lr{lr}", True)
            else:
                models.models.callback_list = checkpoint.checkpoint_callback("efficientYnetV2B0-pretrained", True)
            return EfficientNetV2B0(
                include_top=False,
                weights="imagenet",
                input_shape=models.models.input_shape
            )
    else:
        if eff_type == "S":
            print(F"Setting efficientYnetS")
            models.models.input_shape = (384, 384, 3)
            models.models.callback_list = checkpoint.checkpoint_callback("efficientYnetS", True)
            return EfficientNetV2S(
                include_top=False,
                weights=None,
                input_shape=models.models.input_shape
            )
        elif eff_type == "M":
            print(F"Setting efficientYnetM")
            models.models.input_shape = (480, 480, 3)
            models.models.callback_list = checkpoint.checkpoint_callback("efficientYnetM", True)
            return EfficientNetV2M(
                include_top=False,
                weights=None,
                input_shape=models.models.input_shape
            )
        elif eff_type == "L":
            print(F"Setting efficientYnetL")
            models.models.input_shape = (480, 480, 3)
            models.models.callback_list = checkpoint.checkpoint_callback("efficientYnetL", True)
            return EfficientNetV2L(
                include_top=False,
                weights=None,
                input_shape=models.models.input_shape
            )
        elif eff_type == "B0":
            print(F"Setting efficientYnetB0")
            models.models.input_shape = (224, 224, 3)
            models.models.callback_list = checkpoint.checkpoint_callback("efficientYnetB0", True)
            return EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=models.models.input_shape
            )
        elif eff_type == "V2B0":
            print(F"Setting efficientYnetV2B0")
            models.models.input_shape = (224, 224, 3)
            models.models.callback_list = checkpoint.checkpoint_callback("efficientYnetV2B0", True)
            return EfficientNetV2B0(
                include_top=False,
                weights=None,
                input_shape=models.models.input_shape
            )


# U-net like connections
def set_unet(efficientnet_model, eff_type):
    # Edit efficientNet
    x = efficientnet_model.output
    x = GlobalAveragePooling2D(name='block8_gap')(x)
    x = Dense(256, activation='relu', name='block8_dense')(x)
    prediction = Dense(1, activation='sigmoid', name='prediction')(x)
    if eff_type == "S":
        # Blocks for V2S
        block6_output = efficientnet_model.get_layer('block6o_project_bn').output
        block5_output = efficientnet_model.get_layer('block5i_project_bn').output
        block4_output = efficientnet_model.get_layer('block4f_project_bn').output
        block3_output = efficientnet_model.get_layer('block3d_project_bn').output
        block2_output = efficientnet_model.get_layer('block2d_project_bn').output
        block1_output = efficientnet_model.get_layer('block1b_project_bn').output
        rescaling_output = efficientnet_model.get_layer('rescaling').output

        x = block6_output
        x = upscale_block(512, x, block5_output)
        x = upscale_block(256, x, block3_output)
        x = upscale_block(128, x, block2_output)
        x = upscale_block(64, x, block1_output)
        x = upscale_block(32, x, rescaling_output)
        x = Conv2D(1, (1, 1), padding='same')(x)
        reconstruction = Activation('sigmoid', name="reconstruction")(x)

        return Model(inputs=efficientnet_model.input, outputs=[prediction, reconstruction])
    elif eff_type == "M":
        # Blocks for V2M
        block7_output = efficientnet_model.get_layer('block7e_project_bn').output
        block6_output = efficientnet_model.get_layer('block6r_project_bn').output
        block5_output = efficientnet_model.get_layer('block5n_project_bn').output
        block4_output = efficientnet_model.get_layer('block4g_project_bn').output
        block3_output = efficientnet_model.get_layer('block3e_project_bn').output
        block2_output = efficientnet_model.get_layer('block2e_project_bn').output
        block1_output = efficientnet_model.get_layer('block1c_project_bn').output
        rescaling_output = efficientnet_model.get_layer('rescaling').output

        # x = efficientnet.get_layer('top_activation').output
        x = block7_output
        x = upscale_block(512, x, block5_output)
        x = upscale_block(256, x, block3_output)
        x = upscale_block(128, x, block2_output)
        x = upscale_block(64, x, block1_output)
        x = upscale_block(32, x, rescaling_output)
        x = Conv2D(1, (1, 1), padding='same')(x)
        reconstruction = Activation('sigmoid', name="reconstruction")(x)

        return Model(inputs=efficientnet_model.input, outputs=[prediction, reconstruction])
    elif eff_type == "L":
        # Blocks for V2L
        block7_output = efficientnet_model.get_layer('block7g_project_bn').output
        block6_output = efficientnet_model.get_layer('block6y_project_bn').output
        block5_output = efficientnet_model.get_layer('block5s_project_bn').output
        block4_output = efficientnet_model.get_layer('block4j_project_bn').output
        block3_output = efficientnet_model.get_layer('block3g_project_bn').output
        block2_output = efficientnet_model.get_layer('block2g_project_bn').output
        block1_output = efficientnet_model.get_layer('block1d_project_bn').output
        rescaling_output = efficientnet_model.get_layer('rescaling').output

        # x = efficientnet.get_layer('top_activation').output
        x = block7_output
        x = upscale_block(512, x, block5_output)
        x = upscale_block(256, x, block3_output)
        x = upscale_block(128, x, block2_output)
        x = upscale_block(64, x, block1_output)
        x = upscale_block(32, x, rescaling_output)
        x = Conv2D(1, (1, 1), padding='same')(x)
        reconstruction = Activation('sigmoid', name="reconstruction")(x)

        return Model(inputs=efficientnet_model.input, outputs=[prediction, reconstruction])
    elif eff_type == "B0":
        # Blocks for B0
        block7_output = efficientnet_model.get_layer('block7a_project_bn').output
        block6_output = efficientnet_model.get_layer('block6d_project_bn').output
        block5_output = efficientnet_model.get_layer('block5c_project_bn').output
        block4_output = efficientnet_model.get_layer('block4c_project_bn').output
        block3_output = efficientnet_model.get_layer('block3b_project_bn').output
        block2_output = efficientnet_model.get_layer('block2b_project_bn').output
        block1_output = efficientnet_model.get_layer('block1a_project_bn').output
        rescaling_output = efficientnet_model.get_layer('rescaling').output

        # x = efficientnet.get_layer('top_activation').output
        x = block7_output
        x = upscale_block(512, x, block5_output)
        x = upscale_block(256, x, block3_output)
        x = upscale_block(128, x, block2_output)
        x = upscale_block(64, x, block1_output)
        x = upscale_block(32, x, rescaling_output)
        x = Conv2D(1, (1, 1), padding='same')(x)
        reconstruction = Activation('sigmoid', name="reconstruction")(x)

        return Model(inputs=efficientnet_model.input, outputs=[prediction, reconstruction])
    elif eff_type == "V2B0":
        # Blocks for V2B0
        block6_output = efficientnet_model.get_layer('block6h_project_bn').output
        block5_output = efficientnet_model.get_layer('block5e_project_bn').output
        block4_output = efficientnet_model.get_layer('block4c_project_bn').output
        block3_output = efficientnet_model.get_layer('block3b_project_bn').output
        block2_output = efficientnet_model.get_layer('block2b_project_bn').output
        block1_output = efficientnet_model.get_layer('block1a_project_bn').output
        rescaling_output = efficientnet_model.get_layer('rescaling').output

        x = block6_output
        x = upscale_block(512, x, block5_output)
        x = upscale_block(256, x, block3_output)
        x = upscale_block(128, x, block2_output)
        x = upscale_block(64, x, block1_output)
        x = upscale_block(32, x, rescaling_output)
        x = Conv2D(1, (1, 1), padding='same')(x)
        reconstruction = Activation('sigmoid', name="reconstruction")(x)

        return Model(inputs=efficientnet_model.input, outputs=[prediction, reconstruction])


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


def build_model(trained, eff_type="M", frozen=None, lr=0.0001, eff_weights=None):
    # Efficient net has build in preprocessing
    generators.generators.preprocessing_f = None

    # B0, v2B0, S, M, L
    efficientnet_model = get_efficientnet(trained, eff_type, frozen, lr)
    if frozen is not None:
        for layer in efficientnet_model.layers:
            layer.trainable = False
            print(f"Layer {layer.name} frozen")
            if layer.name == frozen:
                break
    model = set_unet(efficientnet_model, eff_type)
    if eff_weights is not None:
        model.load_weights(eff_weights, by_name=True)
        for layer in model.layers:
            if layer.name.startswith('block') or layer.name.startswith("stem") or layer.name == "input_1" or layer.name == "rescaling" or layer.name == "prediction":
                layer.trainable = False
    print("done")

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss={
            'prediction': 'binary_crossentropy',
            'reconstruction': dice_coef_loss
        },
        metrics={
            'prediction': 'accuracy',
            'reconstruction': 'accuracy'
        }
    )
    model.summary()
    print(model.get_layer("activation_9").get_config())

    return model


if __name__ == "__main__":
    build_model(False)
