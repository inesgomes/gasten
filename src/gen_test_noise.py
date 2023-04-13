import argparse
import os
from dotenv import load_dotenv
from utils import create_and_store_z, gen_seed, set_seed


if __name__ == '__main__':
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--nz", dest="nz", required=True, type=int)
    parser.add_argument("--z-dim", dest="z_dim", required=True, type=int)
    parser.add_argument("--out-dir", dest="out_dir", help="Config file", default=f"{os.environ['FILESDIR']}/data/z")

    args = parser.parse_args()
    seed = gen_seed() if args.seed is None else args.seed

    set_seed(seed)

    test_noise, test_noise_path = create_and_store_z(
        args.out_dir, args.nz, args.z_dim,
        config={'seed': seed, 'n_z': args.nz, 'z_dim': args.z_dim})

    print("Generated test noise, stored in", test_noise_path)
