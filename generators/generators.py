"""
File name: generators.py
Author: Tomas Lapsansky (xlapsa00@stud.fit.vutbr.cz)
Description: Submodule main file.
"""

train_flow = None
valid_flow = None
test_flow = None
val_labels = None
test_labels = None
train_steps = None
valid_steps = None
test_steps = None

is_set = False

dataset_name = ""
epochs = 10

preprocessing_f = None

batch_size = 12
