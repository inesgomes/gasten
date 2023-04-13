import os
import torch
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dotenv import load_dotenv

from src.metrics import fid
from src.metrics.fid import get_inception_feature_map_fn
from src.utils.checkpoint import construct_classifier_from_checkpoint
from src.datasets import get_mnist, get_fashion_mnist, get_cifar10
from src.datasets.utils import BinaryDataset


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', dest='dataroot',
                    default=f"{os.environ['FILESDIR']}/data", help='Dir with dataset')
parser.add_argument('--dataset', dest='dataset',
                    default='fashion-mnist', help='Dataset (mnist or fashion-mnist)')
parser.add_argument('--pos', dest='pos_class', default=3,
                    type=int, help='Positive class for binary classification')
parser.add_argument('--neg', dest='neg_class', default=0,
                    type=int, help='Negative class for binary classification')
parser.add_argument('--batch-size', type=int, default=64,
                    help='Batch size to use')
parser.add_argument('--model-path', dest='model_path', default=None, type=str,
                    help=('Path to classifier to use'
                          'If none, uses InceptionV3'))
parser.add_argument('--num-workers', type=int, default=6)
parser.add_argument('--device', type=str, default='cuda:1',
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--name', dest='name', default=None,
                    help='name of gen .npz file')


def main():
    load_dotenv()
    args = parser.parse_args()
    print(args)

    device = torch.device('cpu' if args.device is None else args.device)
    print("Using device", device)

    num_workers = 0 if args.num_workers is None else args.num_workers

    print("Num workers", num_workers)

    if args.dataset == 'mnist':
        dset = get_mnist(args.dataroot)
    elif args.dataset == 'fashion-mnist':
        dset = get_fashion_mnist(args.dataroot)
    elif args.dataset == 'cifar10':
        dset = get_cifar10(args.dataroot)
    else:
        print("invalid dataset", args.dataset)
        exit(-1)

    if args.model_path is None:
        model_name = 'inception'
    else:
        model_name = '.'.join(os.path.basename(
            args.model_path).split('.')[:-1])

    name = 'stats.{}.{}'.format(
        model_name, args.dataset) if args.name is None else args.name

    binary_classification = args.pos_class is not None and args.neg_class is not None
    if binary_classification:
        name = '{}.{}v{}'.format(name, args.pos_class, args.neg_class)
        dset = BinaryDataset(dset, args.pos_class, args.neg_class)

    print('Dataset size', len(dset))

    dataloader = torch.utils.data.DataLoader(
        dset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    if args.model_path is None:
        get_feature_map_fn, dims = get_inception_feature_map_fn(device)
    else:
        if not os.path.exists(args.model_path):
            print("Model Path doesn't exist")
            exit(-1)
        model = construct_classifier_from_checkpoint(args.model_path)[0]
        model.to(device)
        model.eval()
        model.output_feature_maps = True

        def get_feature_map_fn(images, batch):
            return model(images, batch)[1]

        dims = get_feature_map_fn(dset.data[0:1], (0, 1)).size(1)

    m, s = fid.calculate_activation_statistics_dataloader(
        dataloader, get_feature_map_fn, dims=dims, device=device)

    with open(os.path.join(args.dataroot, 'fid-stats', '{}.npz'.format(name)), 'wb') as f:
        np.savez(f, mu=m, sigma=s)


if __name__ == '__main__':
    main()
