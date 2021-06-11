import numpy as np
import shutil
import math

"""This module is just used to devide the data in dataset/person and dataset/robot
into subsets of training, validation and test data.

We do not use inbuilt functions, because we do want the augmented files to be in the same
set as the original one to avoid dependencies.
"""

# Originally all data was stored in the folders dataset/person and dataset/robot respectiveley
# Normally we could just use SubsetRandomSampler to divide the file into train, val and test set.
# This is not a good idea here, since the files are not indipendent (there are augmented files)
# and we want to have the augmented files in the same set as the original one to 
# avoid dependencys of test, train and val set.

N_OFFLINE_AUGMENTATIONS = 6 # the number of offline augmentations plus 1 (for no augmentation)

indices = np.random.permutation(120)+1

n = 120 # the number of original datapoints (unaugmented) for each class
train_range = math.ceil(n*0.65)

val_range = train_range + math.ceil(n*0.15)

train_indices = indices[:train_range]
val_indices = indices[train_range:val_range]
test_indices = indices[val_range:]

def move_to_folder(indices, folder):
    for label in ['person', 'robot']:
        for i in indices:
            base_name_orig = f"dataset/{label}/{label}{i}"
            base_name_new = f"dataset/{folder}/{label}/{label}{i}"
            shutil.move(f"{base_name_orig}.png", f"{base_name_new}.png")
            for augment in range(1, N_OFFLINE_AUGMENTATIONS):
                ext = f"_aug{augment}.png"
                shutil.move(f"{base_name_orig}{ext}", f"{base_name_new}{ext}")
                
move_to_folder(train_indices, 'train')
move_to_folder(val_indices, 'val')
move_to_folder(test_indices, 'test')