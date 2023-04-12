# gans
Generative Adversarial Net Stuff

## Pre-run

1. run FID
- gen_pairwise_inception.py:

``python stg/gen_pairwise_inception.py``
- specific, instead of pairwise

``python -m stg.metrics.fid --data data/ --dataset mnist --pos 7 --neg 1 --device cpu``

2. gen classifiers
- gen_classifiers.py

``python stg/gen_classifiers.py --pos 7 --neg 1 --nf 1,2,4 --epochs 1``

3. generate test noise
- gen_test_noise.py

``python stg/gen_test_noise.py --nz 2000 --z_dim 64``

## RUN

``python -m stg --config experiments/mnist_7v1.yml``