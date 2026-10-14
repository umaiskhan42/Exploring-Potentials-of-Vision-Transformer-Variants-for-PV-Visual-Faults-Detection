# Vision Transformer variants for multi-label PV defect classification (PVEL-AD)

Code, training logs and split-reproduction script for the manuscript

> F. K. Palijo, I. U. Khalil, M. Ahsan, U. Khan, N. Ullah, S. Misak, L. Prokop,
> *Exploring Potentials of Vision Transformer Variants for PV Visual Faults Detection* (submitted to *Scientific Reports*).

Five ImageNet-pretrained transformers (ViT, Swin, DeiT, PVTv2, BEiT) are fine-tuned under one protocol for **multi-label** classification of electroluminescence (EL) images of PV cells. Each image is labelled with every defect category that appears in its bounding-box annotations.

## Data

The images come from **PVEL-AD** (B. Su, Z. Zhou, H. Chen, *IEEE Trans. Ind. Inform.* 19(1), 404–413, 2023, [doi:10.1109/TII.2022.3162846](https://doi.org/10.1109/TII.2022.3162846)).
The dataset is distributed by its creators on request. To get it, submit the signed request form described at <https://github.com/binyisu/PVEL-AD>.
This repository does **not** contain any PVEL-AD images or annotation files, and they must not be added to it.

The experiments use the annotated `EL2021/trainval` partition (`JPEGImages/` + Pascal-VOC `Annotations/`).

## Contents

| Path | What it is |
|---|---|
| `notebooks/vision_transformer_el_pv.ipynb` | Training and evaluation notebook (Google Colab), with the original per-epoch output of the reported run |
| `results/history_<model>.csv` | Per-epoch train loss, validation loss, label-wise accuracy, macro F1, precision and recall (100 epochs × 5 models), extracted from the notebook output |
| `results/summary.csv` | Final-epoch metrics (Table 4 of the paper), peak macro F1, and means over epochs 91–100 |
| `scripts/dataset_summary.py` | Regenerates the exact image list, category set and 80/20 split from the PVEL-AD annotation files, and prints the per-category counts |
| `scripts/extract_logs_from_notebook.py` | How `results/history_*.csv` were extracted from the notebook output |
| `scripts/summarize_results.py` | Builds `results/summary.csv` from the histories |
| `requirements.txt` | Package versions of the reported run |

## Protocol (as run)

| Item | Setting |
|---|---|
| Images | PVEL-AD `EL2021/trainval` files with an XML annotation, sorted by name, capped at 10,000 |
| Labels | multi-hot vector over the K categories found in the annotations (alphabetical order) |
| Split | `train_test_split(test_size=0.2, random_state=42)`: 80 % train / 20 % validation at image level, not stratified, no separate test set |
| Input | resize 224×224, RGB, ImageNet mean/std normalisation |
| Augmentation | random horizontal flip, random vertical flip (train only) |
| Models (timm 1.0.22, default pretrained weights) | `vit_base_patch16_224`, `swin_tiny_patch4_window7_224`, `deit_tiny_patch16_224`, `pvt_v2_b2`, `beit_base_patch16_224` |
| Head / loss | linear layer with K outputs, `BCEWithLogitsLoss`, prediction threshold 0.5 |
| Optimisation | AdamW, lr 1e-4, batch 16, 100 epochs, no scheduler, no early stopping, no class weights |
| Hardware | Google Colab, NVIDIA T4, Python 3.12, PyTorch 2.9.0 (CUDA 12.6) |

## Results (validation set, epoch 100)

| Model | Label-wise acc. (%) | Val. loss | Macro F1 | Macro recall | Macro precision |
|---|---|---|---|---|---|
| ViT-B/16 | 95.83 | 0.1361 | 0.6305 | 0.6713 | 0.6121 |
| Swin-T | 97.50 | 0.1061 | 0.6731 | 0.6381 | 0.7154 |
| DeiT-Ti | 97.00 | 0.1211 | 0.6450 | 0.6315 | 0.6630 |
| **PVTv2-B2** | **98.58** | **0.0854** | **0.7195** | **0.7028** | **0.7404** |
| BEiT-B | 96.17 | 0.1033 | 0.6519 | 0.6307 | 0.7071 |

Notes:

- **Accuracy** is label-wise (Hamming) accuracy over image–category pairs. Most categories are absent from most images, so it is high even for weak models; **macro F1 is the primary metric**.
- **Macro metrics** use `zero_division=0`, so a category with no positive validation image and no positive prediction counts as 0.
- **Single run** per model, evaluated on a validation split, with no held-out test set.

## Reproducing

1. Obtain PVEL-AD from its providers and place `EL2021/trainval/{JPEGImages,Annotations}` on Google Drive.
2. Open `notebooks/vision_transformer_el_pv.ipynb` in Colab, set `IMG_DIR` / `ANN_DIR` in cell 1, and run all cells.
3. To check the image list and split without training, run:

   ```bash
   pip install scikit-learn
   python scripts/dataset_summary.py <path>/EL2021/trainval/Annotations
   ```

   This prints the dataset size, the categories and the per-category counts, and writes `splits/train_files.txt` and `splits/val_files.txt`.

Training is not seeded beyond the data split, so re-runs will differ slightly from the logged values.
