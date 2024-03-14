# GASTeN Project

![License](https://img.shields.io/static/v1?label=license&message=CC-BY-NC-ND-4.0&color=green)

Variation of GANs that, given a model, generates realistic data that is classiﬁed with low conﬁdence by a given classiﬁer. Results show that the approach is able to generate images that are closer to the frontier when compared to the original ones, but still realistic. Manual inspection conﬁrms that some of those images are confusing even for humans.

Paper: [GASTeN: Generative Adversarial Stress Test Networks](https://link.springer.com/epdf/10.1007/978-3-031-30047-9_8?sharing_token=XGbq9zmVBDFAEaM4r1AAp_e4RwlQNchNByi7wbcMAY55SAL6inraGCkI72KOuzssTzewKWv51v_1pft7j7WJRbiAzL0vaTmG2vf4gs1QhnZ3lV72H7zSKLWQESXZjq5-1pg77WEnt2EHZaN2b51chvHsO6TW3tiGXSVhUgy87Ts%3D)

## Create Virtual Environment

```ssh
mamba create -n gasten python=3.10

mamba env create -f environment.yml
```
*this environment is expectiong cuda=11.8*

## Run

### env file

Create .env file with the following information
```yaml
CUDA_VISIBLE_DEVICES=0
FILESDIR=<file directory>
HOMEDIR=<<file directory>>
ENTITY=<wandb entity to track experiments>
```
HOMEDIR and FILESDIR are different if we save data files in a different place from the repository.

### Preparation

| Step | Description                                                   | command                                                                |
|------|---------------------------------------------------------------|------------------------------------------------------------------------|
| 1    | create FID score for all pairs of numbers                     | `python src/gen_pairwise_inception.py`                                   |
| 1.1  | run for one pair only (e.g. 1vs7)                             | `python -m src.metrics.fid --dataset mnist --pos 7 --neg 1` |
| 2    | create binary classifiers given a pair of numbers (e.g. 1vs7) | `python src/gen_classifiers.py --dataset mnist --pos 7 --neg 1 --nf 1,2,4 --epochs 1`    |
| 3    | create test noise                                             | `python src/gen_test_noise.py --nz 2048 --z-dim 64`                      |

*nz minimum value must be 2048 for FID calculation*

### GASTeN

Change gasten/experiments files according to newly generated data, more specifically, the classifiers name:
- train[step-2][classifier]

Run GASTeN to create images in the boundary between **1** and **7**.

`python -m src --config experiments/mnist_7v1.yml`

### Interpretability module

Run saliency_maps:

`python -m src.xai.saliency_maps --type gasten --no 5`
`python -m src.xai.saliency_maps --type vae --no 7`
`python -m src.xai.saliency_maps --type test`

### Clustering module

- generate images and embeddings (step 1)
    - run_id based on previously trained GASTeN. Check run_id in wandb (as job_name)

`python -m src.clustering.generate_embeddings --config experiments/patterns/mnist_7v1.yml --run_id 0hvkl8kz --epoch 10 --acd_threshold=0.1 --save`
`python -m src.clustering.generate_embeddings --config experiments/patterns/mnist_5v3.yml --run_id ? --epoch 10 --acd_threshold=0.1 --save`
`python -m src.clustering.generate_embeddings --config experiments/patterns/mnist_8v0.yml --run_id ? --epoch 10 --acd_threshold=0.1 --save`
`python -m src.clustering.generate_embeddings --config experiments/patterns/mnist_9v4.yml --run_id ? --epoch 10 --acd_threshold=0.1 --save`

- Testing dimensionality reduction and clustering pairs 
    - dataset ID comes from generate embeddings
    - suggested values come from paper

`python -m src.clustering.test --config experiments/patterns/mnist_7v1.yml --run_id 0hvkl8kz --dim_red umap_80 --clustering gmm_d3`
`python -m src.clustering.test --config experiments/patterns/mnist_5v3.yml --run_id ? --dim_red umap_10 --clustering gmm_s4`
`python -m src.clustering.test --config experiments/patterns/mnist_8v0.yml --run_id ? --dim_red umap_80 --clustering gmm_s3`
`python -m src.clustering.test --config experiments/patterns/mnist_9v4.yml --run_id ? --dim_red umap_80 --clustering gmm_s3`

- Clustering optimization for one dimensionality reduction / clustering technique pair:

`python -m src.clustering.optimize --config experiments/patterns/mnist_7v1.yml --run_id 0hvkl8kz --epoch 10`
`python -m src.clustering.optimize --config experiments/patterns/mnist_5v3.yml --run_id ? --epoch 10`
`python -m src.clustering.optimize --config experiments/patterns/mnist_8v0.yml --run_id ? --epoch 10`
`python -m src.clustering.optimize --config experiments/patterns/mnist_9v4.yml --run_id ? --epoch 10`
