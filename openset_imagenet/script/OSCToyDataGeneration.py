# Create data loaders for toy data experiment
# create a data loader for:
# - train and val (emnist digits + first 13 letters): simply transform labels of letters to -1 (negatives)
# - test (emnist digits + last 13 letters): labels -2 (unknowns)

import torch, torchvision
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
import os
import shutil

# set path parameters
EMNIST_SOUCRCE_PATH = '/local/scratch/bergh/'
TARGET_DATA_PATH = '/local/scratch/bergh/OSCToyData'
# DEVANAGARI_SOURCE_PATH = '/local/scratch/bergh/DevanagariHandwrittenCharacterDataset/Test'
# DEVANAGARI_TARGET_PATH = os.path.join(TARGET_DATA_PATH, 'test')


def create_new_dataset(dataset, data_path, dataset_name, label_map):
    """data will get stored as data_path/dataset_name/original_label/index.png, where index is just an increasing integer."""
    # create target directory if it does not exist
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if os.path.exists(data_path):
        if not os.path.exists(os.path.join(data_path, dataset_name)):
            os.mkdir(os.path.join(data_path, dataset_name))

    dictionary_list = []

    # keep track of filenames (indices) already used per label
    # dict of lists, keys=label, value=list of integers that are used as filenames
    filenumbers_per_label = {}
    
    print(f'processing ({dataset_name})')
    for img, label in tqdm(dataset):
        # give figure some name, i.e., number
        if label in filenumbers_per_label.keys():
            filenumber = filenumbers_per_label[label][-1] +1
            filenumbers_per_label[label].append(filenumber)
        else:
            filenumber = 0
            filenumbers_per_label[label] = [filenumber]
        
        # print(f'label: {label}')
        
        filename = f'{filenumber}' + '.png'
        filepath = os.path.join(dataset_name, f'{label}')

        # save figure
        if not os.path.exists(os.path.join(data_path, filepath)):
            os.mkdir(os.path.join(data_path, filepath))
        img = transforms.functional.rotate(img, 90)
        img.save(os.path.join(data_path, filepath, filename))

        # add figure path/name to labels (path/name should be original_label/filename.png)
        dictionary_list.append({'file': os.path.join(filepath, filename), 'label':label_map[label]})
        labels_df = pd.DataFrame.from_dict(dictionary_list)

    return labels_df


# def move_and_add_devanagari_data(source_path, destination_path, labels_df):
#     """
#     add devanagari test dataset to test data and labels

#     1. get list of foldernames (without digits), these are also the classnames
#     2. for each folder, go through all images and:
#     1. move them to test folder of OSCToyData, but keep folder structure the same
#     2. add path to labels_df_test
#     """
#     dirs = [d for d in os.listdir(source_path) if (not os.path.isfile(d) and d.startswith('character'))]
#     dictionary_list = []

#     # move each directory to OSCToysData/test
#     for directory in tqdm(dirs):
#         if not os.path.exists(os.path.join(destination_path, directory)):
#             os.mkdir(os.path.join(destination_path, directory))
#         files = os.listdir(os.path.join(source_path, directory))
#         for f in files:
#             source = os.path.join(source_path, directory, f)
#             destination = os.path.join(destination_path, directory, f)
#             if os.path.isfile(source):
#                 shutil.copy(source, destination)

#                 # add file to the labels_df_test with label -2
#                 dictionary_list.append({'file': os.path.join('test', directory, f), 'label': -2})
    
#     labels_df_unk = pd.DataFrame.from_dict(dictionary_list)
#     return pd.concat([labels_df, labels_df_unk], ignore_index=True)


def main():

    # --------------------------------------------------
    # define a map from original labels to new labels with negatives

    kn_labels = list(range(10))  # digits [0-9]
    neg_labels = list(range(10, 10+13))  # first 13 letters [A-M]
    unk_labels = list(range(10+13, 10+13+13))  # last 13 letters [N-Z]

    # map original labels to new labels
    label_map = {}

    for i in kn_labels:
        label_map[i] = i

    for i in neg_labels:
        label_map[i] = -1

    for i in unk_labels:
        label_map[i] = -2



    # --------------------------------------------------
    # load EMNIST data from torchvision

    # needs to be transformed, for some reason it is all flipped (and rotated, see later)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(1)
    ])

    # load training and test sets
    digits_all    = torchvision.datasets.EMNIST(root=EMNIST_SOUCRCE_PATH, split='mnist',   train=True, transform=transform, download=True)
    letters_all   = torchvision.datasets.EMNIST(root=EMNIST_SOUCRCE_PATH, split='letters', train=True, transform=transform, download=True)
    digits_test   = torchvision.datasets.EMNIST(root=EMNIST_SOUCRCE_PATH, split='mnist',   train=False, transform=transform, download=True)
    letters_test  = torchvision.datasets.EMNIST(root=EMNIST_SOUCRCE_PATH, split='letters', train=False, transform=transform, download=True)

    # change targets from letters from [1-26] to [10,35]
    letters_all.targets += 9
    letters_test.targets += 9

    # take first half of letters as negatives (i.e., [A-M])
    first_half_labels  = list(range(10,23))
    first_half_indices   = [idx for idx, target in enumerate(letters_all.targets) if target in first_half_labels]
    negatives = torch.utils.data.Subset(letters_all, first_half_indices)

    # split negatives train and validation sets
    knowns_train,  knowns_val = torch.utils.data.random_split(digits_all, [.8, .2])
    negatives_train, negatives_val = torch.utils.data.random_split(negatives, [.8, .2])

    # join them to create new final train, val, test sets
    train_ds = torch.utils.data.ConcatDataset([knowns_train, negatives_train])
    val_ds   = torch.utils.data.ConcatDataset([knowns_val,   negatives_val])
    test_ds  = torch.utils.data.ConcatDataset([digits_test,  letters_test])


    print(f"""
    Dataset sizes:
        - train: {len(train_ds)}
        - val:   {len(val_ds)}
        - test:  {len(test_ds)}
    """)

    print(f"""
    labels (as classes):
          - digits:  {digits_all.classes}
          - digits:  {digits_all.train_labels.unique()}
          - letters: {letters_all.classes}
          - letters: {letters_all.train_labels.unique()}
    """)


    print(f"nr original labels (test): {len(digits_test.classes) + len(letters_test.classes)}")
    print(f"nr labels that are mapped: {len(label_map.keys())}")
    new_labels = []
    [new_labels.append(label) for _, label in label_map.items()]
    new_labels = np.unique(new_labels)
    print(f"nr new labels: {len(new_labels)}")
    print(f"new labels: {new_labels}")



    # --------------------------------------------------
    # # create new dataset
    # iterate over all train, val, and test samples, save them as png and create labels.csv file.
    labels_df_train = create_new_dataset(train_ds, TARGET_DATA_PATH, 'train', label_map)
    labels_df_val   = create_new_dataset(val_ds,   TARGET_DATA_PATH, 'val', label_map)
    labels_df_test  = create_new_dataset(test_ds,  TARGET_DATA_PATH, 'test', label_map)

    # # move devanagari data to test folder and add it to labels_df_test (for labels_test.csv)
    # labels_df_test = move_and_add_devanagari_data(DEVANAGARI_SOURCE_PATH, DEVANAGARI_TARGET_PATH, labels_df_test)

    # save label_train.csv etc without column names
    labels_df_train.to_csv(os.path.join(TARGET_DATA_PATH, 'labels_train.csv'), index=False, header=False)
    labels_df_val.to_csv(  os.path.join(TARGET_DATA_PATH, 'labels_val.csv'),   index=False, header=False)
    labels_df_test.to_csv( os.path.join(TARGET_DATA_PATH, 'labels_test.csv'),  index=False, header=False)


if __name__ == '__main__':
    main()