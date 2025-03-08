import torch


def get_localized_gaussians(gaussians, localization_center, max_distance, localization_type: str = "sphere"):
    if localization_type not in ["sphere", "box"]:
        raise ValueError("Invalid localization type. Must be either 'sphere' or 'box'.")
    
    if localization_type == "sphere":
        distances = torch.linalg.norm(gaussians[:, :3] - localization_center, dim=1)
        localized_gaussians = gaussians[distances < max_distance]
        del distances
    else:
        min_bound = localization_center - max_distance
        max_bound = localization_center + max_distance
        localized_gaussians = gaussians[
            (gaussians[:, 0] > min_bound[0])
            & (gaussians[:, 0] < max_bound[0])
            & (gaussians[:, 1] > min_bound[1])
            & (gaussians[:, 1] < max_bound[1])
            & (gaussians[:, 2] > min_bound[2])
            & (gaussians[:, 2] < max_bound[2])
        ]
    
    del gaussians
    return localized_gaussians
