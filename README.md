# Causal Explanation Fidelity (CEF) – Reproducible PyTorch Pipeline

This repository provides a **working implementation** of the CEF framework described in your manuscript:
*Causal Explanation Fidelity: Bridging Human Reasoning and Model Interpretability in Skin Cancer Diagnosis.*
It includes training for baseline models, saliency generation (Grad‑CAM++ / Attention Rollout), causal perturbations, and unified CEF computation.

> **Note on architectures:** To keep the code self‑contained and light, `InceptionResNetV2+SAB` is provided as a **surrogate** using a ResNet50 backbone plus a CBAM‑style **SAB** attention module. EfficientNet‑B4, ViT‑B/16, and ConvNeXt‑Tiny use torchvision implementations. This does not change the CEF pipeline logic.

---

## 1) Environment

```bash
python -m venv .venv && source .venv/bin/activate  # (Linux/macOS)
pip install -r requirements.txt
```

- Python 3.9+
- PyTorch ≥ 2.1, Torchvision ≥ 0.16

---

## 2) Data Layout

Point `--data_root` to a folder with **patient‑level splits** (70/10/20 suggested) in this structure:

```
DATA_ROOT/
  train/
    class_0/ img1.jpg ...
    class_1/ ...
  val/
    class_0/ ...
    class_1/ ...
  test/
    class_0/ ...
    class_1/ ...
```

For **HAM10000** (7 classes) and **ISIC 2020** (binary), you can export images into the structure above.
All images are automatically resized to **224×224** and normalized to **[0,1]**.

---

## 3) Train Baselines

```bash
# EfficientNet-B4
python train.py --data_root /path/to/DATA_ROOT --model efficientnet_b4 --epochs 100 --batch_size 16 --lr 1e-4 --out checkpoints

# ViT-B/16
python train.py --data_root /path/to/DATA_ROOT --model vit_b16 --epochs 100 --batch_size 16 --lr 1e-4 --out checkpoints

# ConvNeXt-Tiny
python train.py --data_root /path/to/DATA_ROOT --model convnext_tiny --epochs 100 --batch_size 16 --lr 1e-4 --out checkpoints

# InceptionResNetV2+SAB (surrogate)
python train.py --data_root /path/to/DATA_ROOT --model inceptionresnetv2_sab --epochs 100 --batch_size 16 --lr 1e-4 --out checkpoints
```

Each command saves `checkpoints/<model>_best.pt` (by validation accuracy).

> Use `--pretrained` if you want ImageNet weights and **your environment allows downloading** via torchvision.
> For strictly offline training, omit `--pretrained` (default).

---

## 4) Compute CEF

```bash
python evaluate_cef.py   --data_root /path/to/DATA_ROOT   --model efficientnet_b4   --weights checkpoints/efficientnet_b4_best.pt   --batch_size 8   --steps 10
```

- Generates Grad‑CAM++ (CNNs) or Attention‑Rollout (ViT).
- Builds **deletion** and **insertion** curves over `steps` p‑levels.
- Normalizes AUCs with **random** and **uniform** baselines.
- Prints **CEF mean ± std** over the test set.

Repeat for each model to reproduce model‑wise comparisons.

---

## 5) Files & Modules

```
cef_pipeline/
  data/
    datasets.py         # FolderDataset with (train/val/test) class folders
    transforms.py       # Augmentations and resizing
  models/
    efficientnet_b4.py
    vit_b16.py
    convnext_tiny.py
    inceptionresnetv2_sab.py  # ResNet50 + CBAM-like SAB surrogate
  xai/
    gradcam_pp.py       # Grad-CAM++ (generic)
    attention_rollout.py# ViT attention rollout
  cef/
    perturbations.py    # deletion/insert ops + top-mass mask
    cef_metric.py       # CEF computation with baseline normalization
  train.py              # Training loop (Adam, CE loss, early-stop friendly)
  evaluate_cef.py       # CEF evaluation pipeline
  compute_metrics.py    # PAI (IoU) + CAS proxy (optional)
  utils.py              # misc helpers
  requirements.txt
  run_all.sh            # example script
```

---

## 6) Notes & Tips

- **Patient-level splits:** Ensure your data split prevents patient leakage.
- **Lesion masks (optional):** If you have masks, you can adapt `compute_metrics.py` to compute PAI/CAS on your masks.
- **Speed:** Start with smaller `--epochs` to validate the pipeline, then scale up.
- **Device:** Scripts auto-select CUDA if available, otherwise CPU.

---

## 7) Reproducing Paper Sections

- **Fig. Deletion/Insertion Curves:** Log intermediate `g_del(p)`, `g_ins(p)` in `cef_metric.py` for plotting.
- **Table Comparisons:** Run `evaluate_cef.py` for each model to collect CEF and (optionally) accuracy; aggregate into tables.
- **Qualitative Visuals:** Save CAM overlays by writing the `cams` returned inside `evaluate_cef.py` as images.

---

## 8) Disclaimer

This code is provided for **reproducibility of the CEF framework** and is intentionally kept compact.
The `InceptionResNetV2+SAB` here is a faithful *functional surrogate* for demonstration of the causal pipeline.
Swap in your exact SAB block or InceptionResNetV2 backbone if you prefer—no changes to the CEF code are required.
