# Finding Patterns in Ambiguity

![License](https://img.shields.io/static/v1?label=license&message=CC-BY-NC-ND-4.0&color=green)

## Create Virtual Environment

```ssh
conda env create -f environment.yml
```

## Prepare env file

Create .env file with the following information
```yaml
CUDA_VISIBLE_DEVICES=0
FILESDIR=<file directory>
HOMEDIR=<repo directory>
ENTITY=<wandb entity to track experiments>
```
HOMEDIR and FILESDIR are equal if our repository and file directory are the same

**all experiments are saved on [weights&biases](https://wandb.ai/home), so please add your entity**

## Available Datasets

- MNIST
- Fashion MNIST

## Run

### Preparation

**Train binary classifiers**:

`python src/gen_classifiers.py --dataset mnist --pos 7 --neg 1 --nf 1,2,4 --epochs 1`

### Step 1: Synthetic Data Generation

In this module, we generate synthetic data close to the decision boundary using GASTeN: variation of GANs that, given a model, generates realistic data that is classiﬁed with low conﬁdence by a given classiﬁer. Results show that the approach can generate images closer to the frontier than the original ones but still realistic. Manual inspection conﬁrms that some of those images are confusing even for humans.

Paper: [GASTeN: Generative Adversarial Stress Test Networks](https://link.springer.com/epdf/10.1007/978-3-031-30047-9_8?sharing_token=XGbq9zmVBDFAEaM4r1AAp_e4RwlQNchNByi7wbcMAY55SAL6inraGCkI72KOuzssTzewKWv51v_1pft7j7WJRbiAzL0vaTmG2vf4gs1QhnZ3lV72H7zSKLWQESXZjq5-1pg77WEnt2EHZaN2b51chvHsO6TW3tiGXSVhUgy87Ts%3D)

**Train GASTeN**:

1. Prepare configuration file:
    - go to `experiments/original`
    - select the experiment, e.g. `mnist_7v1.yml`
    - change parameters as needed
    - make sure to update the classifier name in `train[step-2][classifier]`
3. Prepare FID score calculation for all pairs of numbers: `python src/gen_pairwise_inception.py`
    - optionally select only one subset: `python -m src.metrics.fid --dataset mnist --pos 7 --neg 1`
4. create test noise: `python src/gen_test_noise.py --nz 2048 --z-dim 64`
   - nz minimum value must be 2048 for FID calculation: 
5. Run GASTeN to create images in the boundary between two classes:  `python -m src --config experiments/original/mnist_7v1.yml`
    - the *original* folder contains the GASTeN paper experiments

### Step 2 & 3: Finding Patterns in Ambiguity & Prototype Selection

This module includes experiments to deep clustering and find prototypes.

1. Prepare configuration file:
   - go to `experiments/clustering`
   - select the experiment, e.g. `mnist_7v1.yml`
   - change parameters as needed, but take into consideration the following:
         - *run_id* is based on previously trained GASTeN. Check run_id in wandb (as job_name).
         - select the GAN epoch that seems to have lower FID and ACD scores
2. Run: `python -m src src.clustering --config experiments/patterns/mnist_7v1.yml`

It is possible to run the experiments sequentially:

1. generate synthetic images and embeddings: `python -m src.clustering.generate_embeddings --config experiments/patterns/mnist_7v1.yml`
2. clustering hyperparameter optimization: `python -m src.clustering.optimize --config experiments/patterns/mnist_7v1.yml`
3. prototype selection: `python -m src.clustering.prototypes --config experiments/patterns/mnist_7v1.yml`



