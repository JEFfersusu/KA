# KA
**Official PyTorch implementation of "Kolmogorov–Arnold Guided Local–Global Attention for Medical Image Classification"**

## Table 1. Summary of Classification and Segmentation Datasets

| Task | Dataset | Image Count | Resolution |
|---|---|---:|:---:|
| Classification | CPN X-ray| 5,228 (3 classes) | 224×224 |
| Classification | PAD-UFES-20| 2,298 (7 classes) | 224×224 |
| Classification | PneumoniaMNIST [b16] | 5,856 (2 classes) | 224×224 |
| Segmentation | BUSI| 647 (breast tumors) | 256×256 |
| Segmentation | GlaS| 165 (gland masks) | 512×512 |
| Segmentation | CVC-ClinicDB| 612 (polyps) | 256×256 |


## Table 2. Performance Comparison of Baseline and Enhanced KA Models

| Model | BUSI IoU (%) | BUSI F1 (%) | GlaS IoU (%) | GlaS F1 (%) | CVC IoU (%) | CVC F1 (%) | Params (M) | GFLOPs|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UNet | 62.20 ± 0.57 | 75.44 ± 0.35 | 87.83 ± 0.19 | 93.37 ± 0.24 | 84.76 ± 0.35 | 91.28 ± 0.32 | 32.79 | 57.23 |
| **UNet + KA** | **65.20 ± 0.61** | **78.57 ± 0.41** | **89.50 ± 0.19** | **94.48 ± 0.24** | 85.69 ± 0.46 | 92.25 ± 0.51 | 32.94 | 57.29 |
| SMAFormer | 51.37 ± 0.23 | 67.94 ± 0.62 | 68.80 ± 0.38 | 81.24 ± 0.37 | 49.94 ± 0.40 | 65.95 ± 0.23 | 216.99 | 31.82 |
| **SMAFormer + KA** | 52.83 ± 0.40 | 68.99 ± 0.39 | 69.23 ± 0.50 | 81.61 ± 0.59 | 53.07 ± 0.34 | 68.33 ± 0.25 | 217.06 | 31.82 |
| UNeXt | 56.70 ± 0.40 | 72.35 ± 0.38 | 70.81 ± 0.82 | 83.27 ± 0.41 | 33.29 ± 0.33 | 49.23 ± 0.30 | **0.25** | **0.11** |
| **UNeXt + KA** | 57.92 ± 0.28 | 73.18 ± 0.41 | 83.04 ± 0.73 | 90.45 ± 0.24 | 64.03 ± 0.35 | 77.69 ± 0.73 | 0.27 | **0.11** |
| MT-UNet | 55.47 ± 0.49 | 70.95 ± 0.55 | 77.92 ± 0.61 | 87.16 ± 0.15 | 43.37 ± 0.63 | 59.71 ± 0.30 | 79.14 | 57.81 |
| **MT-UNet + KA** | 61.71 ± 0.52 | 75.63 ± 0.48 | 78.54 ± 0.40 | 87.88 ± 0.53 | 76.83 ± 0.49 | 87.03 ± 0.22 | 79.28 | 57.81 |
| Rolling-UNet | 48.55 ± 0.57 | 64.85 ± 0.20 | 89.02 ± 0.84 | 94.18 ± 0.37 | 86.35 ± 0.28 | 92.21 ± 0.78 | 7.12 | 8.52 |
| **Rolling-UNet + KA** | 49.43 ± 0.34 | 65.32 ± 0.40 | 88.70 ± 0.45 | 94.20 ± 0.56 | **86.76 ± 0.90** | **92.60 ± 0.39** | 7.16 | 8.79 |


## Table 3. Ablation Study of KLAM and KAM on CVC Dataset (UNeXt)

| KLAM | KAM | IoU (%) | F1 (%) | Params (M) | GFLOPs |
|:---:|:---:|---:|---:|---:|---:|
| ✗ | ✗ | 33.44 | 49.37 | **0.25** | **0.11** |
| ✓ | ✗ | 62.90 | 76.73 | 0.27 | **0.11** |
| ✗ | ✓ | 63.43 | 77.20 | 0.27 | **0.11** |
| ✓ | ✓ | **63.97** | **77.59** | 0.27 | **0.11** |

## Training Iterations Analysis

![Baseline models without KA and training iterations after KA addition on three segmentation datasets. From left to right are Glas, Busi, and CVC datasets.](src/iteration.jpg)

**Figure 1.** Baseline models without KA and training iterations after KA addition on three segmentation datasets. From left to right are Glas, Busi, and CVC datasets.


## Grad-CAM Heatmap Visualization

![Heatmaps for visualization based on Grad-CAM. The visualized layers are all from the last layer before entering the classification head.](src/heatmap.jpg)

**Figure 2.** Heatmaps for visualization based on Grad-CAM. The visualized layers are all from the last layer before entering the classification head.

