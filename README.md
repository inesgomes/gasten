# GASTeN Project

Variation of GANs that, given a model, generates realistic data that is classiﬁed with low conﬁdence by a given classiﬁer. Results show that the approach is able to generate images that are closer to the frontier when compared to the original ones, but still realistic. Manual inspection conﬁrms that some of those images are confusing even for humans.

Paper: [GASTeN: Generative Adversarial Stress Test Networks](https://link.springer.com/epdf/10.1007/978-3-031-30047-9_8?sharing_token=XGbq9zmVBDFAEaM4r1AAp_e4RwlQNchNByi7wbcMAY55SAL6inraGCkI72KOuzssTzewKWv51v_1pft7j7WJRbiAzL0vaTmG2vf4gs1QhnZ3lV72H7zSKLWQESXZjq5-1pg77WEnt2EHZaN2b51chvHsO6TW3tiGXSVhUgy87Ts%3D)

## Create Virtual Environment

```ssh
mamba create -n gasten python=3.10

mamba activate gasten

mamba install pip-tools

pip3 install -r requirements.txt
```

## Run

### env file

Create .env file with the following information
```yaml
CUDA_VISIBLE_DEVICES=0
FILESDIR=<file directory>
ENTITY=<wandb entity to track experiments>
```

### Preparation

| Step | Description                                                   | command                                                                |
|------|---------------------------------------------------------------|------------------------------------------------------------------------|
| 1    | create FID score for all pairs of numbers                     | `python src/gen_pairwise_inception.py`                                   |
| 1.1  | run for one pair only (e.g. 1vs7)                             | `python -m src.metrics.fid --data data/ --dataset mnist --pos 7 --neg 1` |
| 2    | create binary classifiers given a pair of numbers (e.g. 1vs7) | `python src/gen_classifiers.py --pos 7 --neg 1 --nf 1,2,4 --epochs 1`    |
| 3    | create test noise                                             | `python src/gen_test_noise.py --nz 2000 --z-dim 64`                      |

### GASTeN

Run GASTeN to create images in the bounday between **1** and **7**.

`python -m src --config experiments/mnist_7v1.yml`
