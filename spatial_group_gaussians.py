import torch
import math
from collections import deque


def coarse_bin_points(points: torch.Tensor, B: int, device_or_rank):
    """
    Coarse-bins 3D points into a B^3 grid.
    
    Args:
        points (torch.Tensor): 3D coordinates of size (N, 3).
        B (int): Number of bins along each dimension.
        
    Returns:
        counts (torch.Tensor): 1D tensor of length B^3, where counts[i] 
                               is the number of points in bin i.
        bin_indices (torch.Tensor): 1D tensor of length N; bin_indices[j] is 
                                    the linear bin index (0..B^3-1) for points[j].
        bb_min (torch.Tensor): (3,) bounding box minimum.
        bb_max (torch.Tensor): (3,) bounding box maximum.
    """
    N = points.size(0)

    # -- 1. Bounding box
    bb_min = points.min(dim=0).values
    bb_max = points.max(dim=0).values

    # Avoiding degenerate case where bb_max == bb_min
    eps = 1e-8
    ranges = (bb_max - bb_min).clamp(min=eps)  

    # -- 2. Computing the size of each bin along each axis
    side = ranges / B

    # -- 3. Mapping each point (x,y,z) to an integer bin index in [0, B-1]
    bin_coords = ((points - bb_min) / side).floor().long()
    bin_coords = bin_coords.clamp_(min=0, max=B-1)

    # -- 4. Converting the 3D bin (ix,iy,iz) to a single linear index: ix + iy*B + iz*(B^2)
    bin_indices = (bin_coords[:, 0]
                   + bin_coords[:, 1] * B
                   + bin_coords[:, 2] * (B**2))

    # -- 5. Accumulating the count of Gaussians in each bin
    counts = torch.zeros(B**3, dtype=torch.long)
    counts = counts.to(device_or_rank)
    ones = torch.ones(N, dtype=torch.long)
    ones = ones.to(device_or_rank)
    counts.scatter_add_(0, bin_indices, ones)

    return counts, bin_indices, bb_min, bb_max


def merge_cells_into_voxels(counts: torch.Tensor, B: int, K: int, device_or_rank):
    """
    Merges B^3 grid cells into disjoint regions (voxels), each with ~K Gaussians.
    
    Args:
        counts (torch.Tensor): 1D tensor of length B^3 with the number of Gaussians in each cell.
        B (int): Number of cells along each dimension (total cells = B^3).
        K (int): Target number of Gaussians per merged voxel (region).
    
    Returns:
        region_id (torch.Tensor): 1D tensor of length B^3, where region_id[i] gives 
                                  the voxel index for cell i.
        num_voxels (int): The total number of voxels created.
    """
    # region_id[i] = which voxel index cell i belongs to (-1 = unassigned)
    region_id = -1 * torch.ones(B**3, dtype=torch.long)
    region_id = region_id.to(device_or_rank)

    # Function to get (x,y,z) from a linear index
    def idx_to_xyz(idx):
        x = idx % B
        y = (idx // B) % B
        z = idx // (B**2)
        return x, y, z

    # Opposite: get linear index from (x,y,z)
    def xyz_to_idx(x, y, z):
        return x + y * B + z * (B**2)

    # Return valid 6-connected neighbors in 3D
    def get_neighbors(ix, iy, iz):
        neighbors = []
        for (dx, dy, dz) in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            nx, ny, nz = ix + dx, iy + dy, iz + dz
            if (0 <= nx < B) and (0 <= ny < B) and (0 <= nz < B):
                neighbors.append((nx, ny, nz))
        return neighbors

    current_voxel_index = 0

    # Loop over all cells; if unassigned, grow a region from it
    for cell_idx in range(B**3):
        if region_id[cell_idx] != -1:
            continue  # already assigned to some voxel

        # Start a BFS/region from this cell
        queue = deque([cell_idx])
        region_id[cell_idx] = current_voxel_index
        region_count = counts[cell_idx].item()  # how many Gaussians so far

        # Grow until we reach or slightly exceed K
        while queue and region_count < K:
            front = queue.popleft()
            fx, fy, fz = idx_to_xyz(front)
            
            # Check neighbors
            for nx, ny, nz in get_neighbors(fx, fy, fz):
                neighbor_idx = xyz_to_idx(nx, ny, nz)
                if region_id[neighbor_idx] == -1:
                    # Assign neighbor to the same region
                    region_id[neighbor_idx] = current_voxel_index
                    region_count += counts[neighbor_idx].item()
                    queue.append(neighbor_idx)

                    if region_count >= K:
                        # We can stop growing this region now
                        break
            if region_count >= K:
                break
        
        # Done growing this region
        current_voxel_index += 1

    num_voxels = current_voxel_index
    return region_id, num_voxels


def get_padded_tensor(original_tensor, K):
    N, C = original_tensor.shape
    assert K > N, "K must be greater than N"
    
    expanded_tensor = original_tensor.clone()
    
    while expanded_tensor.shape[0] < K:
        for i in range(N):
            if expanded_tensor.shape[0] >= K:
                break
            new_row = original_tensor[N - 1 - i]
            expanded_tensor = torch.cat((expanded_tensor, new_row.unsqueeze(0)), dim=0)
    
    return expanded_tensor


def nearest_neighbor_path(centers):
    """
    centers: list of (center_id, x, y, z)
             e.g. [(id0, x0, y0, z0), (id1, x1, y1, z1), ...]
    
    Returns: a list of center_ids in the order they are visited.
    """
    # 1. Find index of center with the minimum z value
    #    (If multiple centers have the same z, min(...) picks the first it finds)
    start_index = min(range(len(centers)), key=lambda i: centers[i][3])
    
    # 2. Initialize
    n = len(centers)
    visited = [False] * n
    visited[start_index] = True
    
    # The path (ordered list of visited center IDs)
    visited_order = [centers[start_index][0]]
    current_index = start_index
    
    # 3. Iteratively pick the nearest unvisited neighbor
    for _ in range(n - 1):
        nearest_idx = None
        nearest_dist = float('inf')
        
        current_x, current_y, current_z = centers[current_index][1:]
        
        for i in range(n):
            if not visited[i]:
                # Compute Euclidean distance from the current center
                x, y, z = centers[i][1], centers[i][2], centers[i][3]
                dist = math.dist((current_x, current_y, current_z), (x, y, z))
                
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_idx = i
        
        # Mark chosen center, update current
        visited[nearest_idx] = True
        visited_order.append(centers[nearest_idx][0])
        current_index = nearest_idx
    
    # 4. Return the path
    return visited_order

def get_voxelized_gaussians(gaussians_tensor, num_bins, gaussians_per_voxel, tolerance, device_or_rank):
    """
    Voxelizes a tensor of 3D Gaussians.
    """
    gaussian_positions = gaussians_tensor[:, :3].clone().detach()
    counts, bin_indices, bb_min, bb_max = coarse_bin_points(gaussian_positions, num_bins, device_or_rank)
    region_id, num_voxels = merge_cells_into_voxels(counts, num_bins, gaussians_per_voxel, device_or_rank)

    voxelized_gaussians_ids = {}
    for i in range(gaussian_positions.shape[0]):
        bin_id = bin_indices[i].item()
        voxel_id = region_id[bin_id].item()
        if voxel_id not in voxelized_gaussians_ids:
            voxelized_gaussians_ids[voxel_id] = [i]
        else:
            voxelized_gaussians_ids[voxel_id].append(i)
    
    voxelized_gaussians = {}
    for voxel_id, gaussian_ids in voxelized_gaussians_ids.items():
        if len(gaussian_ids) < int(gaussians_per_voxel * tolerance):
            continue
        voxel_gaussians = gaussians_tensor[gaussian_ids]
        gaussians_in_voxels = voxel_gaussians[:gaussians_per_voxel]
        gaussians_in_voxels = gaussians_in_voxels.to(dtype=gaussians_tensor.dtype)
        if len(gaussian_ids) < gaussians_per_voxel:
            gaussians_in_voxels = get_padded_tensor(gaussians_in_voxels, gaussians_per_voxel)
        voxelized_gaussians[voxel_id] = gaussians_in_voxels
    
    return voxelized_gaussians
