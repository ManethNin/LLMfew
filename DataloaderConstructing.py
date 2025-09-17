import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


##This is for building dataset, it can be fed numpy arraies and give the dataloader##
def StringToLabel(y):
    labels = np.unique(y)
    new_label_list = []
    for label in y:
        for position, StringLabel in enumerate(labels):
            if label == StringLabel:
                new_label_list.append(position)
            else:
                continue
    new_label_list = np.array(new_label_list)
    return new_label_list


def get_few_shot_samples(x, y, num_samples_per_class=1):
    class_to_samples = defaultdict(list)

    for idx in range(len(y)):
        label = y[idx].item()
        class_to_samples[label].append((x[idx], y[idx]))

    few_shot_samples = []

    for label, samples in class_to_samples.items():
        if len(samples) >= num_samples_per_class:
            few_shot_samples.extend(random.sample(samples, num_samples_per_class))
        else:
            few_shot_samples.extend(samples)

    # 构建结果的tensor
    x_few_shot = torch.stack([sample[0] for sample in few_shot_samples])
    y_few_shot = torch.stack([sample[1] for sample in few_shot_samples])

    return x_few_shot, y_few_shot


def DataloaderConstructing(dataset, batch_size, shuffle=True, pin_memory=True, few_shot=-1):
    dataset_path = ['npydata/' + dataset + "/" + dataset + "_train_x.npy",
                    'npydata/' + dataset + "/" + dataset + "_train_y.npy",
                    'npydata/' + dataset + "/" + dataset + "_test_x.npy",
                    'npydata/' + dataset + "/" + dataset + "_test_y.npy"]
    X_train, y_train, X_test, y_test = np.load(dataset_path[0]), \
        np.load(dataset_path[1]), \
        np.load(dataset_path[2]), \
        np.load(dataset_path[3])
    y_train, y_test = StringToLabel(y_train), StringToLabel(y_test)
    x_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train.squeeze())
    x_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test.squeeze())

    if few_shot > 0:
        x_train, y_train = get_few_shot_samples(x_train, y_train, num_samples_per_class=few_shot)
    deal_train_dataset, deal_test_dataset = TensorDataset(x_train, y_train), \
        TensorDataset(x_test, y_test)
    train_loader, test_loader = DataLoader(dataset=deal_train_dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           pin_memory=pin_memory,
                                           ), \
        DataLoader(dataset=deal_test_dataset,
                   batch_size=batch_size,
                   shuffle=shuffle,
                   pin_memory=pin_memory,
                   )
    return train_loader, test_loader
