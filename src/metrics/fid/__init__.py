import torch
import numpy as np
from .fid_score import calculate_frechet_distance
from .inception import get_inception_feature_map_fn
from .fid_score import load_statistics_from_path, calculate_activation_statistics_dataloader
from ..metric import Metric


class FID(Metric):
    def __init__(self, feature_map_fn, dims, n_images, mu_real, sigma_real, device='cpu', eps=1e-6):
        super().__init__()
        self.feature_map_fn = feature_map_fn
        self.dims = dims
        self.eps = eps
        self.n_images = n_images
        self.pred_arr = np.empty((n_images, dims))
        self.cur_idx = 0
        self.mu_real = mu_real
        self.sigma_real = sigma_real
        self.device = device

    def update(self, images, batch):
        start_idx, batch_size = batch

        with torch.no_grad():
            pred = self.feature_map_fn(images, start_idx, batch_size)

        pred = pred.cpu().numpy()
        self.pred_arr[self.cur_idx:self.cur_idx + pred.shape[0]] = pred
        self.cur_idx += pred.shape[0]

    def finalize(self):
        act = self.pred_arr
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)

        self.result = calculate_frechet_distance(
            mu, sigma, self.mu_real, self.sigma_real, eps=self.eps)

        return self.result

    def reset(self):
        self.pred_arr = np.empty((self.n_images, self.dims))
        self.cur_idx = 0
        self.result = float('inf')
