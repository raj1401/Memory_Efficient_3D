import os
import pickle
import yaml
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from gsplat import rasterization
import math


class MinMaxScaler:
    def __init__(self, min_vals=None, max_vals=None, range_vals=None):
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.range_vals = range_vals
    
    def fit(self, x: torch.Tensor):
        """
        Computes min and max values for each column (last dimension), ensuring shape (1, 1, 14).
        Args:
            x (torch.Tensor): Input tensor of shape (N, 200000, 14)
        """
        # Compute min and max along axis 1 (sequence length) and keep shape (1, 1, 14)
        self.min_vals = x.amin(dim=(0,1), keepdim=True)  # Shape: (1, 1, 14)
        self.max_vals = x.amax(dim=(0,1), keepdim=True)  # Shape: (1, 1, 14)

        # Compute range, ensuring no division by zero
        self.range_vals = self.max_vals - self.min_vals
        self.range_vals[self.range_vals == 0] = 1  # Avoid division by zero

    def transform(self, x):
        """
        Scales input tensor to range [0,1] using stored min-max values.
        Args:
            x (torch.Tensor): Input tensor of shape (any_batch_size, 200000, 14)
        Returns:
            torch.Tensor: Scaled tensor of shape (any_batch_size, 200000, 14)
        """
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("Scaler has not been fitted. Call fit() first.")
        
        return (x - self.min_vals) / self.range_vals

    def inverse_transform(self, x_scaled):
        """
        Reprojects scaled tensor back to original range.
        Args:
            x_scaled (torch.Tensor): Scaled tensor of shape (any_batch_size, 200000, 14)
        Returns:
            torch.Tensor: Original-scale tensor of shape (any_batch_size, 200000, 14)
        """
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("Scaler has not been fitted. Call fit() first.")
        
        return x_scaled * self.range_vals + self.min_vals


def get_camera_info_from_pickle(path_to_dir, scenes_list):
    """
    This function reads the camera information from the pickled files.
    """
    with open(os.path.join(path_to_dir, f"{scenes_list[0]}.pkl"), "rb") as f:
        initial_camera_info = pickle.load(f)
    
    num_viewpoints = len(initial_camera_info.keys())
    WIDTH = initial_camera_info[0]["width"]
    HEIGHT = initial_camera_info[0]["height"]
    img_shape = initial_camera_info[0]["rendered_img"].shape

    all_viewmats = torch.zeros((len(scenes_list), num_viewpoints, 4, 4))
    all_Ks = torch.zeros((len(scenes_list), num_viewpoints, 3, 3))
    all_rendered_imgs = torch.zeros((len(scenes_list), num_viewpoints, img_shape[1], img_shape[2], img_shape[3]), dtype=torch.bfloat16)

    for i, scene in enumerate(scenes_list):
        with open(os.path.join(path_to_dir, f"{scene}.pkl"), "rb") as f:
            camera_info = pickle.load(f)
        
        for j, key in enumerate(camera_info.keys()):
            all_viewmats[i, j] = torch.tensor(camera_info[key]["viewmats"])
            all_Ks[i, j] = torch.tensor(camera_info[key]["Ks"])
            all_rendered_imgs[i, j] = torch.tensor(camera_info[key]["rendered_img"][0], dtype=torch.bfloat16)
    
    return all_viewmats, all_Ks, all_rendered_imgs, WIDTH, HEIGHT


class GSTrainDataset(Dataset):
    def __init__(self, training_GS_tensor_path, pickle_dir_path, scenes_list, train_test_split=0.8, downsample_factor=2):
        self.num_train_scenes = int(train_test_split * len(scenes_list))
        self.training_GS_tensor = torch.load(training_GS_tensor_path, map_location="cpu")[:self.num_train_scenes, :, 1:]
        self.pickle_dir_path = pickle_dir_path
        self.scenes_list = scenes_list
        self.train_test_split = train_test_split
        self.downsample_factor = downsample_factor
        self.set_cam_info_attributes()
        self.scale_data()
    
    def scale_data(self):
        # Scale data
        train_min_vals = torch.load(os.path.join("data", "train_scaler_min_vals.pt"), map_location="cpu")
        train_max_vals = torch.load(os.path.join("data", "train_scaler_max_vals.pt"), map_location="cpu")

        train_range_vals = train_max_vals - train_min_vals
        train_scaler = MinMaxScaler(train_min_vals, train_max_vals, train_range_vals)

        self.training_GS_tensor = train_scaler.transform(self.training_GS_tensor)
    
    def set_cam_info_attributes(self):
        all_cam_info = get_camera_info_from_pickle(self.pickle_dir_path, self.scenes_list)
        self.all_viewmats = all_cam_info[0][:self.num_train_scenes, :, :, :]
        self.all_ks = all_cam_info[1][:self.num_train_scenes, :, :, :]
        self.all_rendered_imgs = all_cam_info[2][:self.num_train_scenes, :, ::self.downsample_factor, ::self.downsample_factor, :]
        self.width = all_cam_info[3] // self.downsample_factor
        self.height = all_cam_info[4] // self.downsample_factor
    
    def __len__(self):
        return self.num_train_scenes

    def __getitem__(self, idx):
        return self.training_GS_tensor[idx], self.all_viewmats[idx], self.all_ks[idx], self.width, self.height, self.all_rendered_imgs[idx]


class GSValDataset(Dataset):
    def __init__(self, training_GS_tensor_path, pickle_dir_path, scenes_list, train_test_split=0.8, downsample_factor=2):
        self.num_train_scenes = int(train_test_split * len(scenes_list))
        self.training_GS_tensor = torch.load(training_GS_tensor_path, map_location="cpu")[self.num_train_scenes:, :, 1:]
        self.pickle_dir_path = pickle_dir_path
        self.scenes_list = scenes_list
        self.train_test_split = train_test_split
        self.downsample_factor = downsample_factor
        self.set_cam_info_attributes()
        self.scale_data()
    
    def scale_data(self):
        # Scale data
        train_min_vals = torch.load(os.path.join("data", "train_scaler_min_vals.pt"), map_location="cpu")
        train_max_vals = torch.load(os.path.join("data", "train_scaler_max_vals.pt"), map_location="cpu")

        train_range_vals = train_max_vals - train_min_vals
        train_scaler = MinMaxScaler(train_min_vals, train_max_vals, train_range_vals)

        self.training_GS_tensor = train_scaler.transform(self.training_GS_tensor)
    
    def set_cam_info_attributes(self):
        all_cam_info = get_camera_info_from_pickle(self.pickle_dir_path, self.scenes_list)
        self.all_viewmats = all_cam_info[0][self.num_train_scenes:, :, :, :]
        self.all_ks = all_cam_info[1][self.num_train_scenes:, :, :, :]
        self.all_rendered_imgs = all_cam_info[2][self.num_train_scenes:, :, ::self.downsample_factor, ::self.downsample_factor, :]
        self.width = all_cam_info[3] // self.downsample_factor
        self.height = all_cam_info[4] // self.downsample_factor

    
    def __len__(self):
        return len(self.scenes_list) - self.num_train_scenes

    def __getitem__(self, idx):
        return self.training_GS_tensor[idx], self.all_viewmats[idx], self.all_ks[idx], self.width, self.height, self.all_rendered_imgs[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000, dtype=torch.bfloat16, device="cuda:0"):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.dtype = dtype

        pe = torch.zeros(max_len, d_model, dtype=dtype, device=device)
        position = torch.arange(0, max_len, dtype=dtype, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=dtype, device=device) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(dtype=x.dtype)


class AttentionDownscaler(nn.Module):
    def __init__(self, d_model=14, nhead=2, num_layers=3, dim_feedforward=64, 
                 dropout=0.1, chunk_size=7500, overlap=2000, conv_kernel_size=500,
                 initial_length=30000, target_len=15000, dtype=torch.bfloat16, device="cuda:0", scaler=None):
        super(AttentionDownscaler, self).__init__()

        self.chunk_size = chunk_size
        self.overlap = overlap  # Overlap size
        self.stride = chunk_size - overlap  # Non-overlapping step
        self.initial_length = initial_length
        self.conv_kernel_size = conv_kernel_size
        self.target_len = target_len
        self.dtype = dtype
        self.device = device
        self.scaler = scaler

        self.pos_encoder = PositionalEncoding(d_model, max_len=chunk_size, dtype=dtype)

        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, activation="gelu"
            ) for i in range(num_layers)
        ])

        for layer in self.encoder_layers:
            layer.to(dtype)

        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size, stride=2, padding=(conv_kernel_size//2 - 1), dtype=dtype),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(target_len)
        )

        self.attn_downscale = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True, dtype=dtype)

        self.linear_block_1 = nn.Sequential(
            nn.Linear(d_model, d_model, dtype=dtype),
            nn.GELU()
        )

        self.linear_block_2 = nn.Sequential(
            nn.Linear(d_model, d_model, dtype=dtype)
        )


    def forward(self, gaussians_tensor, viewmats, Ks, width, height, inference=False):
        batch_size, seq_len, feature_dim = gaussians_tensor.shape
        assert seq_len >= self.chunk_size, "Sequence length should be at least the chunk size"

        gaussians_tensor = gaussians_tensor.to(self.device, dtype=self.dtype)

        num_chunks = (seq_len - self.overlap) // self.stride
        output_chunks = torch.zeros(batch_size, seq_len, feature_dim, device=self.device, dtype=self.dtype)
        overlap_counts = torch.zeros(batch_size, seq_len, 1, device=self.device, dtype=self.dtype)

        # Process chunks with overlapping
        for i in range(num_chunks):
            start_idx = i * self.stride
            end_idx = min(start_idx + self.chunk_size, seq_len)
            actual_chunk_size = end_idx - start_idx

            chunk = gaussians_tensor[:, start_idx:end_idx, :].clone()
            pos_encoded_chunk = self.pos_encoder.pe[:, : actual_chunk_size, :]
            chunk = chunk + pos_encoded_chunk.to(chunk.dtype)

            for layer_idx, encoder_layer in enumerate(self.encoder_layers):
                chunk = encoder_layer(chunk)

            # Store chunk and count overlaps for averaging
            output_chunks[:, start_idx:end_idx, :] += chunk
            overlap_counts[:, start_idx:end_idx, :] += 1

        # Normalize overlapping regions by averaging
        output_chunks /= overlap_counts.clamp(min=1)

        # Permute for Conv1D (batch, d_model, seq_len)
        gs_out = output_chunks.permute(0, 2, 1)

        # Conv1D
        gs_out = self.conv_block(gs_out)

        # Permute back to (batch, target_len, d_model)
        gs_out = gs_out.permute(0, 2, 1)

        # Attention-based downscaling
        # Perform similar chunk-wise processing
        num_gs_chunks = (gs_out.shape[1] - self.overlap) // self.stride
        output_gs_chunks = torch.zeros(batch_size, gs_out.shape[1], feature_dim, device=self.device, dtype=self.dtype)
        overlap_gs_counts = torch.zeros(batch_size, gs_out.shape[1], 1, device=self.device, dtype=self.dtype)

        # Process chunks with overlapping
        for i in range(num_gs_chunks):
            start_idx = i * self.stride
            end_idx = min(start_idx + self.chunk_size, gs_out.shape[1])
            actual_chunk_size = end_idx - start_idx

            gs_chunk = gs_out[:, start_idx:end_idx, :].clone()
            pos_encoded_gs_chunk = self.pos_encoder.pe[:, : actual_chunk_size, :]
            gs_chunk = gs_chunk + pos_encoded_gs_chunk.to(gs_chunk.dtype)

            gs_chunk = self.attn_downscale(gs_chunk, gs_chunk, gs_chunk)[0]

            # Store chunk and count overlaps for averaging
            output_gs_chunks[:, start_idx:end_idx, :] += gs_chunk
            overlap_gs_counts[:, start_idx:end_idx, :] += 1

        # Normalize overlapping regions by averaging
        output_gs_chunks /= overlap_gs_counts.clamp(min=1)
        
        gs_out += output_gs_chunks

        # Linear
        gs_out = self.linear_block_1(gs_out)
        gs_out = self.linear_block_2(gs_out)

        # Re-scaling
        if self.scaler is not None:
            gs_out = self.scaler.inverse_transform(gs_out)

        # Batch Rendering
        if not inference:
            gs_out = gs_out.to(dtype=torch.float32)

            all_rendered_images = []
            for i in range(batch_size):
                rendered_images, _, _ = rasterization(
                means=gs_out[i, :, :3], colors=gs_out[i, :, 3:6], opacities=gs_out[i, :, 6],
                scales=gs_out[i, :, 7:10], quats=gs_out[i, :, 10:14], viewmats=viewmats[i].to(gs_out.device),
                Ks=Ks[i].to(gs_out.device), width=width[i], height=height[i]
            )
                all_rendered_images.append(rendered_images)

            all_rendered_images = torch.stack(all_rendered_images, dim=0)
            all_rendered_images = all_rendered_images.to(dtype=torch.bfloat16)
            gs_out = gs_out.to(dtype=torch.bfloat16)
        else:
            all_rendered_images = None

        return gs_out, all_rendered_images


def save_checkpoint(epoch, model, optimizer, train_loss_history, val_loss_history, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path} \n")


    # Save loss values as JSON
    loss_file = os.path.join(checkpoint_dir, "train_loss_log.json")
    with open(loss_file, "w") as f:
        json.dump(train_loss_history, f, indent=4)
    
    loss_file = os.path.join(checkpoint_dir, "val_loss_log.json")
    with open(loss_file, "w") as f:
        json.dump(val_loss_history, f, indent=4)


def downscaling_loss(rendered_images, target_images):
    # Process batch
    loss = 0
    for i in range(rendered_images.shape[0]):
        loss += F.mse_loss(rendered_images[i], target_images[i], reduction="mean")
    return loss / rendered_images.shape[0]


def evaluate(model, val_dataloader):
    print("Validation...")
    model.eval()
    total_loss = 0

    for i, (gs_tensor, viewmats, Ks, width, height, target_images) in enumerate(val_dataloader):
        _, rendered_images = model(gs_tensor, viewmats, Ks, width, height)
        loss = downscaling_loss(rendered_images, target_images.to(rendered_images.device))
        total_loss += loss.item()

        print(f"Batch {i+1}/{len(val_dataloader)}, Loss {loss.item():.4f}", end="\r")
    
    avg_loss = total_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_loss:.4f} \n")
    return avg_loss


def train(
        num_layers, dim_feedforward, dropout, chunk_size, overlap, conv_kernel_size, initial_length,
        target_len, batch_size, epochs, learning_rate, training_GS_path, pickle_path, scenes_list,
        train_test_split, rasterization_downsample_factor, checkpoint_interval=2, checkpoint_dir="transformer_checkpoints"
):
    """
    num_layers: int # Number of transformer encoder layers
    dim_feedforward: int # Feedforward dimension
    dropout: float # Dropout rate
    chunk_size: int # Chunk size for processing
    overlap: int # Overlap size between chunks
    conv_kernel_size: int # Conv1D kernel size
    initial_length: int # Initial length of the sequence
    target_len: int # Target length after compression
    batch_size: int
    epochs: int
    learning_rate: float
    training_GS_path: str
    pickle_path: str
    scenes_list: list of str
    train_test_split: float
    rasterization_downsample_factor: int
    checkpoint_interval: int
    """
    
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Scaler for data
    # Scale data
    train_min_vals = torch.load(os.path.join("data", "train_scaler_min_vals.pt"), map_location=DEVICE)
    train_max_vals = torch.load(os.path.join("data", "train_scaler_max_vals.pt"), map_location=DEVICE)

    train_range_vals = train_max_vals - train_min_vals
    train_scaler = MinMaxScaler(train_min_vals, train_max_vals, train_range_vals)

    # Model
    model = AttentionDownscaler(
        num_layers=num_layers, dim_feedforward=dim_feedforward,
        dropout=dropout, chunk_size=chunk_size, overlap=overlap, conv_kernel_size=conv_kernel_size,
        initial_length=initial_length, target_len=target_len,
        device=DEVICE, scaler=train_scaler
    )
    model.to(DEVICE)

    # Loss and Optimizer
    criterion = downscaling_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Dataset and DataLoader
    train_dataset = GSTrainDataset(training_GS_path, pickle_path, scenes_list, train_test_split, rasterization_downsample_factor)
    val_dataset = GSValDataset(training_GS_path, pickle_path, scenes_list, train_test_split, rasterization_downsample_factor)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint")]
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            checkpoint = torch.load(os.path.join(checkpoint_dir, latest_checkpoint), map_location=torch.device("cpu"))

            current_epoch = checkpoint["epoch"]

            model.load_state_dict(checkpoint["model_state_dict"])            
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            train_loss_history = checkpoint["train_loss_history"]
            val_loss_history = checkpoint["val_loss_history"]
            print(f"Loaded checkpoint from {latest_checkpoint}")
    else:
        current_epoch = 0
        train_loss_history = {}
        val_loss_history = {}

    print(f"\n Total number of model parameters: {sum(p.numel() for p in model.parameters())} \n")

    # Training loop
    for epoch in range(current_epoch, epochs):
        print(f"Epoch {epoch + 1}")
        print("---------------------------")

        model.train()
        total_loss = 0

        print("Training...")

        for i, (gs_tensor, viewmats, Ks, width, height, target_images) in enumerate(train_dataloader):
            optimizer.zero_grad()
            _, rendered_images = model(gs_tensor, viewmats, Ks, width, height)
            loss = criterion(rendered_images, target_images.to(rendered_images.device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Batch {i+1}/{len(train_dataloader)} Loss: {loss.item():.4f}", end="\r")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Average Training Loss: {avg_loss:.4f} \n")

        # Store loss for logging
        train_loss_history[epoch + 1] = avg_loss

        # Validation
        val_loss = evaluate(model, val_dataloader)
        val_loss_history[epoch + 1] = val_loss


        # Save checkpoint periodically
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(epoch + 1, model, optimizer, train_loss_history, val_loss_history, checkpoint_dir)

    print("Training complete!")


def main(config_path):
    if torch.cuda.is_available():
        print("CUDA available")
    else:
        print("No GPUs available")
        exit()
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Get name of yml file. This will be used to save checkpoints
    config_filename = os.path.basename(config_path)
    config_filename = os.path.splitext(config_filename)[0]

    print(f"Will save checkpoints to checkpoints/{config_filename}")

    # Load scenes list
    with open("scenes_list.txt", "r") as f:
        scenes_list = f.read().splitlines()
    
    training_GS_path = os.path.join("data", "voxelized_train_tensor.pt")
    pickle_path = os.path.join("data", "Training_Camera_Info")

    num_layers = config["num_layers"]
    dim_feedforward = config["dim_feedforward"]
    dropout = config["dropout"]
    chunk_size = config["chunk_size"]
    overlap = config["overlap"]
    conv_kernel_size = config["conv_kernel_size"]
    initial_length = config["initial_length"]
    target_len = config["target_len"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    train_test_split = config["train_test_split"]
    rasterization_downsample_factor = config["rasterization_downsample_factor"]
    checkpoint_interval = config["checkpoint_interval"]
    checkpoint_dir = os.path.join("transformer_checkpoints", config_filename)

    train(
        num_layers, dim_feedforward, dropout, chunk_size, overlap, conv_kernel_size,
        initial_length, target_len, batch_size, epochs, learning_rate, training_GS_path, pickle_path, scenes_list,
        train_test_split, rasterization_downsample_factor, checkpoint_interval, checkpoint_dir
    )


CONFIG_PATH = "train_config.yml"
main(CONFIG_PATH)