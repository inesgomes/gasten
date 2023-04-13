import itertools
import subprocess
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', dest='dataroot',
                    default=f"{os.environ['FILESDIR']}/data", help='Dir with dataset')
parser.add_argument('--dataset', dest='dataset',
                    default='mnist', help='Dataset (mnist or fashion-mnist or cifar10)')
parser.add_argument('--n-classes', dest='n_classes',
                    default=10, help='Number of classes in dataset')
parser.add_argument('--device', type=str, default='cuda:1',
                    help='Device to use. Like cuda, cuda:0 or cpu')


def main():
    args = parser.parse_args()
    print(args)

    n_classes = args.n_classes

    for neg_class, pos_class in itertools.combinations(range(n_classes), 2):
        print(f"{neg_class}vs{pos_class}")
        proc = subprocess.run(['python', '-m', 'stg.metrics.fid',
                               '--data', args.dataroot,
                               '--dataset', args.dataset,
                               '--device', args.device,
                               '--pos', str(pos_class), '--neg', str(neg_class)])


if __name__ == '__main__':
    main()
