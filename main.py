"""
File name: main.py
Author: Tomas Lapsansky (xlapsa00@stud.fit.vutbr.cz)
Description: This file is used as the main executable script of the project. It takes care of processing the arguments
            and setting the parameters of the model, dataset and then starting the training or evaluation process.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from sklearn import metrics

from generators import generators, kaggle140k, ffc, ffcCNN, celebDF
from models import models, densenet121, vgg19, xception, resnet50, resnet50v2, efficientnet, efficientdet
from custommodels import efficientYnet, efficientYdet


def parse_args():
    parser = argparse.ArgumentParser(description='Running defined models')
    parser.add_argument('-m', action='store', dest='training_model',
                        help='Model name (case sensitive)', required=True, default=None)
    parser.add_argument('-d', action='store', dest='dataset',
                        help='Dataset name (case sensitive)', required=False, default=None)
    parser.add_argument('-e', action='store', dest="eval",
                        help='Set if execute evaluation', required=False, default=None)
    parser.add_argument('-r', action='store', dest="dropout",
                        help='Set dropout rate', required=False, default="0.5")
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


def assign_model(name, trained, eff_type=None, frozen=False, lr=0.0001, checkpoint=None, dropout_rate=0.5):
    if name == "densenet121":
        model = densenet121.build_model(trained, lr)
    elif name == "vgg19":
        model = vgg19.build_model(trained, lr)
    elif name == "xception":
        model = xception.build_model(trained, lr)
    elif name == "resnet50":
        model = resnet50.build_model(trained, lr)
    elif name == "resnet50v2":
        model = resnet50v2.build_model(trained, lr)
    elif name == "efficientnet":
        model = efficientnet.build_model(trained, eff_type, frozen, lr)
    elif name == "efficient-ynet":
        model = efficientYnet.build_model(trained, eff_type, frozen, lr, checkpoint)
    elif name == "efficientdet":
        model = efficientdet.build_model(trained, eff_type, frozen, lr, dropout_rate)
    elif name == "efficientdetBigger":
        model = efficientdet.build_model(trained, eff_type, frozen, lr, dropout_rate, True)
    elif name == "efficient-ydet":
        model = efficientYdet.build_model(trained, eff_type, frozen, lr, dropout_rate)
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
    elif name == "ffc224sn":
        if models.input_shape[0] != 224:
            print("Bad dataset shape", file=sys.stderr)
            exit(1)
        ffcCNN.init(224, True)
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
    elif name == "ffc480n":
        if models.input_shape[0] != 480:
            print("Bad dataset shape", file=sys.stderr)
            exit(1)
        ffc.init(480, True)
    elif name == "ffc480sn":
        if models.input_shape[0] != 480:
            print("Bad dataset shape", file=sys.stderr)
            exit(1)
        ffcCNN.init(480, True)
    elif name == "ffc480sn":
        if models.input_shape[0] != 480:
            print("Bad dataset shape", file=sys.stderr)
            exit(1)
        ffcCNN.init(480, True)
    elif name == "ffc480sn-100":
        if models.input_shape[0] != 480:
            print("Bad dataset shape", file=sys.stderr)
            exit(1)
        ffcCNN.init(480, True, 100)
    elif name == "ffc480sn-80":
        if models.input_shape[0] != 480:
            print("Bad dataset shape", file=sys.stderr)
            exit(1)
        ffcCNN.init(480, True, 80)
    elif name == "ffc480sn-60":
        if models.input_shape[0] != 480:
            print("Bad dataset shape", file=sys.stderr)
            exit(1)
        ffcCNN.init(480, True, 60)
    elif name == "ffc480sn-40":
        if models.input_shape[0] != 480:
            print("Bad dataset shape", file=sys.stderr)
            exit(1)
        ffcCNN.init(480, True, 40)
    elif name == "celeb-df":
        if models.input_shape[0] != 480:
            print("Bad dataset shape", file=sys.stderr)
            exit(1)
        celebDF.init(480)
    elif name == "celeb-df-100":
        if models.input_shape[0] != 480:
            print("Bad dataset shape", file=sys.stderr)
            exit(1)
        celebDF.init(480, 100)
    elif name == "celeb-df-80":
        if models.input_shape[0] != 480:
            print("Bad dataset shape", file=sys.stderr)
            exit(1)
        celebDF.init(480, 80)
    elif name == "celeb-df-60":
        if models.input_shape[0] != 480:
            print("Bad dataset shape", file=sys.stderr)
            exit(1)
        celebDF.init(480, 60)
    elif name == "celeb-df-40":
        if models.input_shape[0] != 480:
            print("Bad dataset shape", file=sys.stderr)
            exit(1)
        celebDF.init(480, 40)


def get_output_index_by_name(model, output_name):
    output_names = [output.name.split("/")[0] for output in model.outputs]
    return output_names.index(output_name)


def print_evaluate(model, image_name):
    print("Evaluation of model using " + generators.dataset_name)

    # output_index = get_output_index_by_name(model, "prediction")

    y_pred = model.predict(generators.test_flow)
    y_test = generators.test_labels

    # print("ROC-AUC Score:", metrics.roc_auc_score(y_test, y_pred[output_index]))
    print("ROC-AUC Score:", metrics.roc_auc_score(y_test, y_pred))
    print("AP Score:", metrics.average_precision_score(y_test, y_pred))
    print()
    print(metrics.classification_report(y_test, y_pred > 0.5))

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred > 0.5)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["Fake", "Real"])

    cm_display.plot()
    plt.savefig("images/" + image_name + ".png")


def main():
    arguments = parse_args()

    # Assign dataset name before model assignment
    generators.dataset_name = arguments.dataset

    model = assign_model(arguments.training_model, arguments.trained, arguments.type, arguments.frozen,
                         float(arguments.learning_rate), arguments.checkpoint, float(arguments.dropout))
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
        y_pred = model.predict(img)
        print(y_pred)
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

        # Plot the training and validation loss
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.title(f"{arguments.training_model} Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        if arguments.frozen is not None:
            plt.savefig(f"{arguments.training_model}_loss.png")
        else:
            plt.savefig(f"{arguments.training_model}_frozen_{arguments.frozen}_loss.png")

        plt.clf()

        # Plot the training and validation accuracy
        plt.plot(history.history['accuracy'], label='train_acc')
        plt.plot(history.history['val_accuracy'], label='val_acc')
        plt.title(f"{arguments.training_model} Accuracy")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        if arguments.frozen is not None:
            plt.savefig(f"{arguments.training_model}_accuracy.png")
        else:
            plt.savefig(f"{arguments.training_model}_frozen_{arguments.frozen}_accuracy.png")


if __name__ == "__main__":
    main()
