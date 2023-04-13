import torch
from .datasets import get_mnist, get_fashion_mnist, get_cifar10
from .utils import BinaryDataset


dataset_2_fn = {
    'mnist': get_mnist,
    'fashion-mnist': get_fashion_mnist,
    'cifar10': get_cifar10,
}


def valid_dataset(name):
    return name.lower() in dataset_2_fn


def load_dataset(name, data_dir, pos_class=None, neg_class=None, train=True):
    if not valid_dataset(name):
        print("{} dataset not supported".format(name))
        exit(-1)

    get_dset_fn = dataset_2_fn[name]
    dataset = get_dset_fn(data_dir, train=train)

    image_size = tuple(dataset.data.shape[1:])
    if len(image_size) == 2:
        image_size = 1, *image_size
    elif len(image_size) == 3:
        if image_size[2] == 3:
            image_size = image_size[2], image_size[0], image_size[1]

    targets = dataset.targets if torch.is_tensor(
        dataset.targets) else torch.Tensor(dataset.targets)

    num_classes = targets.unique().size()

    if pos_class is not None and neg_class is not None:
        num_classes = 2
        dataset = BinaryDataset(dataset, pos_class, neg_class)

    return dataset, num_classes, image_size
