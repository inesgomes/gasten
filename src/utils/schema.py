import os
from schema import Schema, Optional, And, Or
from src.datasets import valid_dataset


CONFIG_SCHEMA_GASTEN = Schema({
    "project": str,
    "name": str,
    "out-dir": os.path.exists,
    "data-dir": os.path.exists,
    "fid-stats-path": os.path.exists,
    "fixed-noise": Or(And(str, os.path.exists), int),
    "test-noise": os.path.exists,
    Optional("compute-fid"): bool,
    Optional("device", default="cpu"): str,
    Optional("num-workers", default=0): int,
    Optional("num-runs", default=1): int,
    Optional("step-1-seeds"): [int],
    Optional("step-2-seeds"): [int],
    "dataset": {
        "name": And(str, valid_dataset),
        Optional("binary"): {"pos": int, "neg": int}
    },
    "model": {
        "z_dim": int,
        "architecture": Or({
            "name": "dcgan",
            "g_filter_dim": int,
            "d_filter_dim": int,
            "g_num_blocks": int,
            "d_num_blocks": int,
        }, {
            "name": "resnet",
            "g_filter_dim": int,
            "d_filter_dim": int,
        }),
        "loss": Or({
            "name": "wgan-gp",
            "args": {
                "lambda": int,
            }
        }, {
            "name": "ns"
        })
    },
    "optimizer": {
        "lr": float,
        "beta1": Or(float, int),
        "beta2": Or(float, int),
    },
    "train": {
        "step-1": Or(And(str, os.path.exists), {
            "epochs": int,
            "checkpoint-every": int,
            "batch-size": int,
            "disc-iters": int,
            Optional("early-stop"): {
                "criteria": int,
            }
        }),
        "step-2": {
            # TODO
            Optional("step-1-epochs", default="best"): [Or(int, "best", "last")],
            Optional("early-stop"): {
                "criteria": int,
            },
            "epochs": int,
            "checkpoint-every": int,
            "batch-size": int,
            "disc-iters": int,
            "classifier": [And(str, os.path.exists)],
            "weight": [Or(int, float, "mgda", "mgda:norm")]
        }
    }
})

CONFIG_SCHEMA_CLUSTERING = Schema({
    "project": str,
    "name": str,
    Optional("device", default="cuda:0"): str,
    "batch-size": int,
    "checkpoint": bool,
    "compute-fid": bool,
    "dir": {
        "data": os.path.exists,
        "clustering": os.path.exists,
        "fid-stats": os.path.exists,
    },
    "dataset": {
        "name": And(str, valid_dataset),
        Optional("binary"): {"pos": int, "neg": int}
    },
    "gasten": {
        "classifier": [And(str, os.path.exists)],
        "run-id": str,
        "weight": int,
        "epoch": {
            "step-1": int,
            "step-2": int
        },
    },
    "clustering":{
      "z-dim": int,
        "fixed-noise": int,
        "acd": float,
        "n-iter": int,
        "options": [{
            "dim-reduction": "umap",  
            "clustering":  Or("hdbscan", "gmm")
        }]
    },
    "prototypes": {
        "type": [Or("medoid", "random")]
    }
})
