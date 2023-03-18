import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from sklearn import metrics

from generators import generators, kaggle140k, dfdc, ffc, ffcCNN
from models import models, densenet121, vgg19, xception, resnet50, resnet50v2, efficientnet
from custommodels import efficientYnet


def parse_args():
    parser = argparse.ArgumentParser(description='Running defined models')
    parser.add_argument('-m', action='store', dest='training_model',
                        help='Model name (case sensitive)', required=True, default=None)
    parser.add_argument('-d', action='store', dest='dataset',
                        help='Dataset name (case sensitive)', required=False, default=None)
    parser.add_argument('-e', action='store', dest="eval",
                        help='Set if execute evaluation', required=False, default=None)
    parser.add_argument('-t', action='store_true', dest="trained",
                        help='Use pre-trained model', required=False, default=False)
    parser.add_argument('-f', action='store', dest="frozen",
                        help='Freeze layers of base model until', required=False, default=None)
    parser.add_argument('--type', action='store', dest="type",
                        help='Set type of efficient net', required=False, default=None)
    parser.add_argument('--lr', action='store', dest="learning_rate",
                        help='Set learning rate for model', required=False, default="0.0001")
    parser.add_argument('-c', action='store', dest="checkpoint",
                        help='Path to loaded checkpoint', required=False, default=None)
    parser.add_argument('-p', action='store', dest="print",
                        help='Path to image', required=False, default=None)

    return parser.parse_args()


def assign_model(name, trained, eff_type=None, frozen=False, lr=0.0001, checkpoint=None):
    if name == "densenet121":
        model = densenet121.build_model(trained)
    elif name == "vgg19":
        model = vgg19.build_model(trained)
    elif name == "xception":
        model = xception.build_model(trained)
    elif name == "resnet50":
        model = resnet50.build_model(trained)
    elif name == "resnet50v2":
        model = resnet50v2.build_model(trained)
    elif name == "efficientnet":
        model = efficientnet.build_model(trained, frozen, lr)
    elif name == "efficient-ynet":
        model = efficientYnet.build_model(trained, eff_type, frozen, lr, checkpoint)
    else:
        return None

    return model


def assign_dataset(name):
    if name == "kaggle-140k":
        kaggle140k.init()
    elif name == "dfdc":
        exit(0)
    elif name == "ffc224":
        if models.input_shape[0] != 224:
            print("Bad dataset shape", file=sys.stderr)
            exit(1)
        ffc.init(224)
    elif name == "ffc384":
        if models.input_shape[0] != 384:
            print("Bad dataset shape", file=sys.stderr)
            exit(1)
        ffc.init(384)
    elif name == "ffc480":
        if models.input_shape[0] != 480:
            print("Bad dataset shape", file=sys.stderr)
            exit(1)
        ffc.init(480)
    elif name == "ffc480s":
        if models.input_shape[0] != 480:
            print("Bad dataset shape", file=sys.stderr)
            exit(1)
        ffcCNN.init(480)


def print_evaluate(model, image_name):
    print("Evaluation of model using " + generators.dataset_name)

    y_pred = model.predict(generators.test_flow)
    y_test = generators.test_flow.classes

    print("ROC-AUC Score:", metrics.roc_auc_score(y_test, y_pred))
    print("AP Score:", metrics.average_precision_score(y_test, y_pred))
    print()
    print(metrics.classification_report(y_test, y_pred > 0.5))

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred > 0.5)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["Fake", "Real"])

    cm_display.plot()
    plt.savefig(image_name + ".png")


def main():
    arguments = parse_args()

    # Assign dataset name before model assignment
    generators.dataset_name = arguments.dataset

    model = assign_model(arguments.training_model, arguments.trained, arguments.type, arguments.frozen, float(arguments.learning_rate), arguments.checkpoint)
    if model is None:
        print("Unresolved model name")
        return 1

    if arguments.print is not None:
        img_data = tf.io.read_file(os.path.abspath(arguments.print))
        img = tf.io.decode_png(img_data)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = np.expand_dims(img, axis=0)
        print(model.input_shape)
        print(img.shape)
        y_pred, y_mask = model.predict(img)
        print(y_pred)
        print(y_mask)
        img = np.uint8(y_mask * 255)
        cv2.imshow("image", y_mask[0])
        cv2.waitKey()
        return 0

    assign_dataset(arguments.dataset)
    if not generators.is_set:
        print("Unresolved dataset name")
        return 1

    if arguments.eval:
        # evaluating model
        checkpoint = arguments.eval
        print("Using checkpoint", checkpoint)
        if not os.path.exists(checkpoint):
            print("Trained model not found")
            return 1
        model.built = True
        model.load_weights(checkpoint)

        print_evaluate(model, os.path.basename(checkpoint))
    else:
        # training model
        history = model.fit(
            generators.train_flow,
            epochs=generators.epochs,
            steps_per_epoch=generators.train_steps,
            validation_data=generators.valid_flow,
            validation_steps=generators.valid_steps,
            callbacks=models.callback_list
        )


if __name__ == "__main__":
    main()
