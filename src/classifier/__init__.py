from .simple_cnn import Classifier as SimpleCNN
from .my_mlp import Classifier as MyMLP
from .classifier_cache import ClassifierCache


def construct_classifier(params, device=None):
    if params['type'] == 'cnn':
        C = SimpleCNN(params['img_size'], params['nf'],
                      params['n_classes'])
    elif params['type'] == 'mlp':
        C = MyMLP(params['img_size'], params['n_classes'], params['nf'])
    else:
        exit(-1)

    return C.to(device)
