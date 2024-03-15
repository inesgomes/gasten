# GASTeN Project

![License](https://img.shields.io/static/v1?label=license&message=CC-BY-NC-ND-4.0&color=green)

Variation of GANs that, given a model, generates realistic data that is classiﬁed with low conﬁdence by a given classiﬁer. Results show that the approach is able to generate images that are closer to the frontier when compared to the original ones, but still realistic. Manual inspection conﬁrms that some of those images are confusing even for humans.

Paper: [GASTeN: Generative Adversarial Stress Test Networks](https://link.springer.com/epdf/10.1007/978-3-031-30047-9_8?sharing_token=XGbq9zmVBDFAEaM4r1AAp_e4RwlQNchNByi7wbcMAY55SAL6inraGCkI72KOuzssTzewKWv51v_1pft7j7WJRbiAzL0vaTmG2vf4gs1QhnZ3lV72H7zSKLWQESXZjq5-1pg77WEnt2EHZaN2b51chvHsO6TW3tiGXSVhUgy87Ts%3D)

## Create Virtual Environment

```ssh
mamba create -n gasten python=3.10

mamba env create -f environment.yml
```
*this environment.yml is expectiong cuda=11.8*

## Run

### env file

Create .env file with the following information
```yaml
CUDA_VISIBLE_DEVICES=0
FILESDIR=<file directory>
HOMEDIR=<repo directory>
ENTITY=<wandb entity to track experiments>
```
HOMEDIR and FILESDIR are equal if our repository and file directory are the same

### Available Datasets

- MNIST
- Fashion MNIST

### Preparation

| Step | Description                                                   | command                                                                |
|------|---------------------------------------------------------------|------------------------------------------------------------------------|
| 1    | create FID score for all pairs of numbers                     | `python src/gen_pairwise_inception.py`                                   |
| optional  | run for one pair only (e.g. 1vs7)                             | `python -m src.metrics.fid --dataset mnist --pos 7 --neg 1` |
| 2    | create binary classifiers given a pair of numbers (e.g. 1vs7) | `python src/gen_classifiers.py --dataset mnist --pos 7 --neg 1 --nf 1,2,4 --epochs 1`    |
| 3    | create test noise                                             | `python src/gen_test_noise.py --nz 2048 --z-dim 64`                      |

*nz minimum value must be 2048 for FID calculation*

### GASTeN

1. Change gasten/(...)/experiments files according to newly generated data, more specifically, the classifiers name:
    - train[step-2][classifier]

2. Run GASTeN to create images in the boundary between two classes:
    - e.g. **7** vs **1**
    - the *original* folder contains the GASTeN paper experiments

    `python -m src --config experiments/original/mnist_7v1.yml`


### Clustering module

This module includes the experiments done to distil the decision boundary.

| Step | Description                                                   | command                                                                |
|------|---------------------------------------------------------------|------------------------------------------------------------------------|
| 1    | generate images and embeddings, for a given ACD threshold     | `python -m src.clustering.generate_embeddings --config experiments/patterns/mnist_7v1.yml --run_id 0hvkl8kz --epoch 10 --acd_threshold=0.1 --save`                                   |
| 2 | Clustering optimization for one dimensionality reduction / clustering technique pair  | `python -m src.clustering.optimize --config experiments/patterns/mnist_7v1.yml --run_id 0hvkl8kz` |
| optional    | Testing dimensionality reduction and clustering pairs |`python -m src.clustering.test --config experiments/patterns/mnist_7v1.yml --run_id 0hvkl8kz --dim_red umap_80 --clustering gmm_d3`    |

Some hints:
- run_id based on previously trained GASTeN. Check run_id in wandb (as job_name).
- select the GAN epoch that seems to yeld lost FID and ACD scores
- the test.py options are available in the dictionary on the python file


### Interpretability module

This module includes some experiments to understand GASTeN synthetically generated images.

To run saliency_maps:

`python -m src.xai.saliency_maps --type gasten --no 5`

`python -m src.xai.saliency_maps --type vae --no 7`

`python -m src.xai.saliency_maps --type test`