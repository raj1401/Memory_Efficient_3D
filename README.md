Utilizing 2D Attention-Based Models for Memory Efficient 3D Reconstruction
==========================================================================

This repository contains the official implementation of my MS Thesis **"Utilizing 2D Attention-Based Models for Memory Efficient 3D Reconstruction"**.

📌 Overview
-----------

3D Gaussian Splatting (3DGS) has recently emerged as a powerful representation for real-time 3D scene reconstruction and rendering. While it provides high visual fidelity and efficiency, it often requires **millions of Gaussians**, leading to **large memory/storage overhead**.

This work introduces a **novel spatial grouping + 2D attention framework** that compresses 3DGS representations while preserving high visual quality. Our method can reduce memory usage by **up to 16×** without significant loss in rendering fidelity.

* * * * *

🚀 Key Contributions
--------------------

-   **Gaussian Importance Ranking**\
    A new pruning strategy ranks Gaussians based on opacity, volume, and luminance, retaining only the most important ones.

-   **Spatial Grouping**\
    Gaussians are rearranged into a structured 2D matrix so that adjacent rows correspond to nearby points in 3D space, enabling effective use of attention mechanisms.

-   **2D Attention-Based Compression Model**\
    A lightweight model combining **Transformer encoders, convolutional downscaling, and pooling** to learn compact scene representations.

-   **Supervised Initialization**\
    Pre-training on curated Gaussian subsets stabilizes convergence and improves training efficiency.

* * * * *

📊 Experimental Results
-----------------------

-   **Dataset**: Evaluated on the CO3D dataset (hydrant category, ~224 scenes).

-   **Compression Ratios**: Achieved **2×, 4×, 8×, and 16×** compression.

-   **Metrics**:

    -   At 16× compression, reconstructions maintain strong fidelity:

        -   PSNR ≈ 33 dB

        -   SSIM ≈ 0.95

        -   LPIPS ≈ 0.07

-   **Qualitative Findings**: Even at **16× compression**, scenes preserve overall geometry and structure, with only minor loss of fine details (e.g., text/smudges on hydrants).

* * * * *

🧩 Methodology
--------------

1.  **3DGS Generation**:\
    Scenes reconstructed with Nerfstudio + COLMAP, producing Gaussian splats.

2.  **Preprocessing & Pruning**:\
    Normalize scene size → prune excess Gaussians → duplicate if under-sampled.

3.  **Spatial Grouping**:\
    Partition space into voxels → merge into regions → reorder Gaussians for locality.

4.  **Compression Model**:

    -   Sliding-window self-attention over Gaussian tokens.

    -   1D convolutions with pooling for downscaling.

    -   Self-attention + FFN for reconstruction and rendering.

5.  **Training**:

    -   Optimizer: Adam

    -   Loss: MSE over rendered images from 8 viewpoints

* * * * *

📈 Results Summary
------------------

| Compression | PSNR (Test) | SSIM | LPIPS |
| --- | --- | --- | --- |
| 2× | 46.4 dB | 0.99 | 0.02 |
| 4× | 38.4 dB | 0.97 | 0.04 |
| 8× | 34.5 dB | 0.96 | 0.06 |
| 16× | 32.4 dB | 0.95 | 0.07 |

* * * * *

🏆 Conclusion
-------------

Our framework demonstrates that **attention-based compression** is highly effective for reducing the size of 3DGS scenes. By leveraging both **short- and long-range dependencies** among Gaussians, we can achieve **up to 16× compression** with negligible degradation in visual quality, enabling scalable and memory-efficient 3D reconstruction for downstream applications such as **AR/VR, robotics, and digital content creation**.
