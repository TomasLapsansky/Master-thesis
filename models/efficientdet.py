import sys

import tensorflow as tf
from keras.layers import BatchNormalization
from keras.regularizers import l2
from tensorflow import keras
from keras import layers, Model

# Import EfficientNetV2M model
from keras.applications import EfficientNetV2M, EfficientNetV2L

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


def create_detection_head(num_classes, num_anchors):
    class_head = layers.Conv2D(num_classes * num_anchors, 1, padding='same')
    box_head = layers.Conv2D(4 * num_anchors, 1, padding='same')
    return class_head, box_head


def create_efficientdet(num_classes, num_anchors_per_scale, input_shape=(512, 512, 3), num_channels=64):
    inputs = layers.Input(shape=input_shape)
    efficientnet = EfficientNetV2M(include_top=False, input_tensor=inputs)

    C2, C3, C4, C5 = efficientnet.get_layer('block2c_add').output, \
        efficientnet.get_layer('block3d_add').output, \
        efficientnet.get_layer('block4f_add').output, \
        efficientnet.get_layer('block5f_add').output

    P3 = create_bifpn_layer(C3, num_channels)([C3])
    P4 = create_bifpn_layer(C4, num_channels)([C4, P3])
    P5 = create_bifpn_layer(C5, num_channels)([C5, P4])
    P6 = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5)

    class_head, box_head = create_detection_head(num_classes, num_anchors_per_scale)
    classes = []
    boxes = []
    for feature_map in [P3, P4, P5, P6]:
        classes.append(class_head(feature_map))
        boxes.append(box_head(feature_map))

    # Resize class and box predictions to have the same spatial dimensions
    target_shape = tf.shape(classes[0])[1:3]
    resized_classes = [classes[0]]
    resized_boxes = [boxes[0]]
    for i in range(1, len(classes)):
        resized_classes.append(tf.image.resize(classes[i], target_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
        resized_boxes.append(tf.image.resize(boxes[i], target_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))

    classes = layers.Concatenate(axis=1)(resized_classes)
    classes = layers.Reshape((-1, num_classes))(classes)
    classes = layers.Activation('sigmoid')(classes)  # Use sigmoid activation for binary classification

    boxes = layers.Concatenate(axis=1)(resized_boxes)
    boxes = layers.Reshape((-1, 4))(boxes)

    outputs = layers.Concatenate(axis=2)([classes, boxes])

    model = Model(inputs=inputs, outputs=outputs)
    return model


def create_efficientdetM_binary_classification_bigger(num_channels, l2_regularization=0.01, dropout_rate=0.5):
    backbone = EfficientNetV2M(include_top=False, input_shape=(480, 480, 3), weights='imagenet')
    # C3, C4, C5 = backbone.layers[-188].output, backbone.layers[-94].output, backbone.layers[-1].output
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
    dense1 = layers.Dense(256, activation='relu', kernel_regularizer=l2(l2_regularization))(pooled_features)
    batch_norm1 = BatchNormalization()(dense1)
    dropout1 = layers.Dropout(dropout_rate)(batch_norm1)

    dense2 = layers.Dense(128, activation='relu', kernel_regularizer=l2(l2_regularization))(dropout1)
    batch_norm2 = BatchNormalization()(dense2)
    dropout2 = layers.Dropout(dropout_rate)(batch_norm2)

    output = layers.Dense(1, activation='sigmoid')(dropout2)

    model = Model(inputs=backbone.input, outputs=output)
    return model


def create_efficientdetM_binary_classification(num_channels, dropout_rate=0.5):
    backbone = EfficientNetV2M(include_top=False, input_shape=(480, 480, 3), weights='imagenet')
    # C3, C4, C5 = backbone.layers[-188].output, backbone.layers[-94].output, backbone.layers[-1].output
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
    output = layers.Dense(1, activation='sigmoid')(dropout)

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

    output = layers.Dense(1, activation='sigmoid')(dropout2)

    model = Model(inputs=backbone.input, outputs=output)
    return model


def create_efficientdetL_binary_classification(num_channels, dropout_rate=0.5):
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
    dense = layers.Dense(128, activation='relu')(pooled_features)
    dropout = layers.Dropout(dropout_rate)(dense)
    output = layers.Dense(1, activation='sigmoid')(dropout)

    model = Model(inputs=backbone.input, outputs=output)
    return model


def build_model(trained, eff_type="M", frozen=None, lr=0.0001, dropout_rate=0.5, robust=False):
    models.models.input_shape = (480, 480, 3)

    generators.generators.preprocessing_f = None
    # num_classes = 1  # Binary classification
    # num_anchors_per_scale = 9
    # efficientdet_model = create_efficientdet(num_classes, num_anchors_per_scale)
    if eff_type == "M":
        if robust:
            efficientdet_model = create_efficientdetM_binary_classification_bigger(num_channels=64, dropout_rate=dropout_rate)
        else:
            efficientdet_model = create_efficientdetM_binary_classification(num_channels=64, dropout_rate=dropout_rate)
    elif eff_type == "L":
        if robust:
            efficientdet_model = create_efficientdetL_binary_classification_bigger(num_channels=64, dropout_rate=dropout_rate)
        else:
            efficientdet_model = create_efficientdetL_binary_classification(num_channels=64, dropout_rate=dropout_rate)
    else:
        print("Bad efficientdet type", file=sys.stderr)
        exit(1)

    if frozen is not None:
        for layer in efficientdet_model.layers:
            layer.trainable = False
            print(f"Layer {layer.name} frozen")
            if layer.name == frozen:
                break

    if frozen is not None:
        if robust:
            models.models.callback_list = checkpoint.checkpoint_callback(f"efficientdetMBigger-f-{frozen}-lr{lr}-dr{dropout_rate}")
        else:
            models.models.callback_list = checkpoint.checkpoint_callback(f"efficientdetM-f-{frozen}-lr{lr}-dr{dropout_rate}")
    else:
        if robust:
            models.models.callback_list = checkpoint.checkpoint_callback(f"efficientdetMBigger-lr{lr}")
        else:
            models.models.callback_list = checkpoint.checkpoint_callback(f"efficientdetM-lr{lr}")

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    # optimizer = keras.optimizers.Adam()
    efficientdet_model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # efficientdet_model.summary()

    return efficientdet_model
