import numpy as np
import torch


def rank_gaussians(gaussians, scale_weight, color_weight, alpha_weight):
    """
    Ranks the gaussians based on their scale, color and alpha.
    """
    vol = torch.prod(gaussians[:, 7:10], axis=1)
    # avg_scale = np.mean(gaussians[:, 7:10], axis=1)
    luminescence = 0.299 * gaussians[:, 3] + 0.587 * gaussians[:, 4] + 0.114 * gaussians[:, 5]
    alphas = gaussians[:, 6]

    importance = scale_weight * vol + color_weight * luminescence + alpha_weight * alphas
    # importance = scale_weight * avg_scale + color_weight * luminescence + alpha_weight * alphas

    min_imp = torch.min(importance)
    max_imp = torch.max(importance)
    denom = max_imp - min_imp if max_imp != min_imp else 1e-9
    normalized_importance = (importance - min_imp) / denom

    sorted_indices = torch.argsort(-normalized_importance)
    return gaussians[sorted_indices]
