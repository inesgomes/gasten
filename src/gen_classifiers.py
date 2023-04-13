import itertools
import subprocess
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
from dotenv import load_dotenv


load_dotenv()
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--data-dir', dest='dataroot',
                    default=f"{os.environ['FILESDIR']}/data", help='Dir with dataset')
parser.add_argument('--out-dir', dest='out_dir',
                    default=f"{os.environ['FILESDIR']}/models", help='Path to generated files')
parser.add_argument('--dataset', dest='dataset',
                    default='mnist', help='Dataset (mnist or fashion-mnist or cifar10)')
parser.add_argument('--n-classes', dest='n_classes',
                    default=10, help='Number of classes in dataset')
parser.add_argument('--device', type=str, default='cpu',
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--batch-size', dest='batch_size',
                    type=int, default=64, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='ADAM opt learning rate')

parser.add_argument('--pos', dest='pos_class', default=9,
                    type=int, help='Positive class for binary classification')
parser.add_argument('--neg', dest='neg_class', default=4,
                    type=int, help='Negative class for binary classification')

parser.add_argument('--epochs', type=str, default="3",
                    help='List of number of epochs to train for')
parser.add_argument('--classifier-type', dest='clf_type',
                    type=str, help='list with elements "cnn" or "mlp"', default='cnn')
parser.add_argument('--nf', type=str, default="2,4,8,16",
                    help='List of possible num features')


def main():
    args = parser.parse_args()
    print(args)

    n_classes = args.n_classes

    l_epochs = list(set([e
                    for e in args.epochs.split(",") if e.isdigit()]))
    l_clf_type = list(set([ct
                           for ct in args.clf_type.split(",")]))
    l_nf = list(set([nf
                    for nf in args.nf.split(",") if nf.isdigit()]))
    l_epochs.sort()
    l_clf_type.sort()
    l_nf.sort()

    if args.pos_class is not None and args.neg_class is not None:
        iterator = iter([(str(args.neg_class), str(args.pos_class))])
    else:
        iterator = itertools.combinations(range(n_classes), 2)

    for neg_class, pos_class in iterator:
        print(f"\nGenerating classifiers for {pos_class}v{neg_class} ...")
        for clf_type, nf, epochs in itertools.product(l_clf_type, l_nf, l_epochs):
            print("\n", clf_type, nf, epochs)
            proc = subprocess.run(["python", "-m", "src.classifier.train",
                                   "--device", args.device,
                                   "--data-dir", args.dataroot,
                                   "--out-dir", args.out_dir,
                                   "--dataset", args.dataset,
                                   "--pos", pos_class,
                                   "--neg", neg_class,
                                   "--classifier-type", clf_type,
                                   "--nf", nf,
                                   "--epochs", epochs,
                                   "--batch-size", str(args.batch_size),
                                   "--lr", str(args.lr)],
                                  capture_output=True)
            for line in proc.stdout.split(b'\n')[-4:-1]:
                print(line.decode())


if __name__ == '__main__':
    main()
