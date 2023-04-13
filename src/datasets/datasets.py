import torchvision


def get_mnist(dataroot, train=True):
    dataset = torchvision.datasets.MNIST(root=dataroot, download=True, train=train,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                 (0.5,), (0.5,)),
                                         ]))

    return dataset


def get_fashion_mnist(dataroot, train=True):
    dataset = torchvision.datasets.FashionMNIST(root=dataroot, download=True,
                                                train=train,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                        (0.5,), (0.5,)),
                                                ]))

    return dataset


def get_cifar10(dataroot, train=True):
    dataset = torchvision.datasets.CIFAR10(root=dataroot, download=True, train=train,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))

    return dataset
