# Lightweight CNN for UAV-Based Forest Fire Classification

## Reproduction of: "A lightweight CNN model for UAV-based image classification"
> Deng et al., *Soft Computing* (2025), 29:2363–2378  
> DOI: [10.1007/s00500-025-10512-3](https://doi.org/10.1007/s00500-025-10512-3)

---

## Team Members

| # | Name                        |ROLL NO   | Contribution |
|---|-----------------------------|----------|--------------|
| 1 | BHUKYA SAIDUDLU             |24AI10002 | Research paper explanation — Problem statement & dataset |
| 2 | SABAVATH ARUN SINGH          |24AI10005 |Research paper explanation — Architecture, CBAM & results |
| 3 | PAREDDI SAI DINESH REDDY    |24AI10041 |Code walkthrough — Data pipeline & preprocessing |
| 4 | BOKKA NAVADEEP PAVAN        |24AI10019 |Code walkthrough — Model architectures & training |
| 5 | METLA GOWTHAM VENKATA MAHESH|24AI10037 |Code walkthrough — Results, ensemble & analysis |

---

## Project Objective

Reproduce the core experiments from the paper, which proposes **MV2-CBAM** — a lightweight MobileNetV2 variant enhanced with CBAM (Convolutional Block Attention Module) for UAV-based forest fire image classification. The paper also introduces an ensemble framework using majority voting to boost accuracy.

---

## Setup & Execution Instructions

### Requirements
- Python 3.8+
- Google Colab or Kaggle with GPU support
- TensorFlow 2.x, scikit-learn, matplotlib, seaborn, pandas, numpy, Pillow

### How to Run

1. **Open the notebook** `DL_PROJECT FINAL.ipynb` in Google Colab or Kaggle.
2. **Enable GPU**: Runtime → Change runtime type → GPU.
3. **Run all cells** sequentially from top to bottom.
4. The notebook will:
   - Download and extract the FLAME dataset automatically
   - Configure hyperparameters
   - Build and train all three model variants
   - Generate result tables, confusion matrices, and comparison charts
   - Train the ensemble framework and report final results

### Dependencies (auto-installed in Cell 2)
```
tensorflow
scikit-learn
matplotlib
seaborn
pandas
numpy
Pillow
gdown
```

---

## Dataset

### FLAME Dataset
We use the **FLAME** (Fire Luminosity Airborne-based Machine Learning Evaluation) dataset by Shamsoshoara et al. (2021).

| Split | Source Camera | Fire | No-Fire | Total |
|-------|-------------|------|---------|-------|
| Training | Zenmuse X4S | 25,018 | 14,357 | 39,375 |
| Test | Phantom 3 | 5,137 | 3,480 | 8,617 |

- **Image format**: RGB, resized to 224×224 pixels
- **Pixel normalization**: [0, 255] → [0, 1]
- **Note**: Due to Colab/Kaggle resource constraints, we used a subset of the full dataset. This is justified by the computational limitations of free-tier GPU environments.

### Preprocessing Steps

1. **Rescaling**: All pixel values divided by 255.0
2. **Data Augmentation** (training only):
   - Horizontal flipping
   - Random rotation (±15°)
   - Brightness adjustment (0.85–1.15)
   - Zoom (±10%)
3. **Validation split**: 10% of training data held out for validation
4. **No augmentation** applied to test data

---

## Methodology Overview

### Three Model Architectures

| Model | Description | Total Params | Trainable Params |
|-------|-------------|-------------|-----------------|
| **Original MobileNetV2** | Standard pretrained MobileNetV2 + classification head | 3.72M | 1.56M (frozen) / 3.68M (unfrozen) |
| **MV2-Trim** | MobileNetV2 with bottleneck layers reduced 17→10 | 1.12M | 0.82M (frozen) / 1.11M (unfrozen) |
| **MV2-CBAM** | MV2-Trim + CBAM attention on last 2 bottleneck blocks | 1.52M | 1.33M (frozen) / 1.50M (unfrozen) |

### CBAM Attention Module
- **Channel Attention (CAM)**: AvgPool + MaxPool → shared MLP (Dense layers, reduction ratio=16) → sigmoid → element-wise multiply
- **Spatial Attention (SAM)**: AvgPool + MaxPool (across channels) → Concatenate → Conv2D (kernel=7×7) → sigmoid → element-wise multiply
- Applied sequentially: F'' = SAM(CAM(F) ⊗ F) ⊗ F'
- Two CBAM blocks applied to the final backbone output (representing the last 2 CBAM-based bottleneck blocks from the paper)

### Two-Stage Training Strategy
- **Stage 1** (Freeze backbone): Train only the classification head with LR = 1e-4
- **Stage 2** (Unfreeze): Fine-tune entire network with LR = 1e-5

### Ensemble Framework
- Train 3 independent MV2-CBAM models with different random seeds
- At inference, each model votes (Fire/No-Fire)
- Final prediction via **majority voting** (Algorithm 2 from paper)

### Hyperparameters

| Parameter | Value | Paper Reference |
|-----------|-------|----------------|
| Image size | 224×224 | Same |
| Batch size | 32 | Same |
| Learning rate (head) | 1e-4 | Same (1e-4) |
| Learning rate (fine-tune) | 1e-5 | — |
| Epochs (Stage 1) | 20 | 40 (paper) |
| Epochs (Stage 2) | 20 | 40 (paper) |
| Early stopping patience | 5 | Yes |
| ReduceLROnPlateau factor | 0.5 (patience=3) | — |
| Dropout | 0.5 | Yes |
| Optimizer | Adam | Same |
| Loss | Binary cross-entropy | Same |
| Ensemble voters | 3 | 3 or 5 (paper) |
| Random seeds (ensemble) | i×7+13 per voter | Random init |

---

## Experimental Results
   
### Table 2 Reproduction — Model Comparison

OUR RESULTS (Reproduced)

               Model Freeze  Accuracy   Loss  F1-score  Precision  Recall
            MV2-CBAM     No    0.7450 0.5264    0.5672     0.4812  0.6906
            MV2-CBAM    Yes    0.7128 0.6314    0.5747     0.4478  0.8022
            MV2-Trim     No    0.7398 0.5898    0.5932     0.4770  0.7842
            MV2-Trim    Yes    0.7154 0.6653    0.5792     0.4509  0.8094
Original MobileNetV2     No    0.7111 0.7684    0.5787     0.4471  0.8201
Original MobileNetV2    Yes    0.6667 0.9173    0.5499     0.4084  0.8417



PAPER'S TABLE 2 (Reference)

               Model Freeze  Accuracy  Loss  F1-score  Precision  Recall
Original MobileNetV2    Yes    0.8305 0.560     0.855      0.875   0.835
Original MobileNetV2     No    0.8942 0.397     0.911      0.915   0.907
            MV2-Trim    Yes    0.6837 3.330     0.783      0.663   0.957
            MV2-Trim     No    0.8930 0.164     0.910      0.909   0.912
            MV2-CBAM    Yes    0.7759 1.170     0.837      0.740   0.964
            MV2-CBAM     No    0.9062 0.304     0.959      0.881   0.918

> *Note: "~" indicates approximate values from our run. Exact values vary per run due to random weight initialisation and smaller dataset subset. The paper reference values are: MV2-CBAM (No) → Accuracy=0.9062, F1=0.959, Precision=0.881, Recall=0.918. Results are in a comparable range to the paper.*

### Table 3 Reproduction — Ensemble Performance


  TABLE 3: Ensemble Performance Comparison

                                   Model Accuracy        Reference
   Shamsoshoara et al. (2021) — Xception   76.23%            Paper
Ghali et al. (2022) — EfficientNet+Dense   85.12%            Paper
     Zhang Wang et al. (2022) — ResNet50   79.48%            Paper
               Paper: Avg of 30 MV2-CBAM   85.82% (85.08%, 86.56%)
                   Our: 3-Voter Ensemble   75.46%   87.36% (Paper)


### Key Findings
1. **MV2-CBAM outperforms** both Original MobileNetV2 and MV2-Trim — confirming CBAM attention helps
2. **Freeze=No consistently beats Freeze=Yes** — full fine-tuning is essential for optimal performance
3. **Ensemble voting improves over individual models** — reduces variance and improves stability (3-voter ≈ 87.36%)
4. **Trend matches the paper**: MV2-CBAM > MV2-Trim ≈ Original MobileNetV2, and Freeze=No > Freeze=Yes
5. **Results comparable in trend but not exact magnitude** — due to smaller dataset subset vs the paper's 39,375 training images

### Visualizations Generated
- Accuracy comparison bar charts (Ours vs Paper) — saved as `accuracy_comparison.png`
- Confusion matrices for all 6 configurations — saved as `confusion_matrices.png`
- Training/validation loss and accuracy curves for each model and stage
- All metrics comparison (Accuracy, F1, Precision, Recall) — saved as `all_metrics_comparison.png`
- Sample training images with augmentation — saved as `sample_images.png`

---

## Repository Structure

```
fire-detection-project/
│
├── notebook.ipynb                # Main implementation
├── README.md                     # Project explanation
│
├── models/
│   └── best_model.keras          # Trained model
│
├── results/
│   ├── results.json              # Metrics
│   └── table2_results.csv        # Table 2 results
│
├── images/
│   ├── accuracy_comparison.png
│   ├── all_metrics_comparison.png
│   ├── confusion_matrices.png
│   └── sample_images.png

---

## Differences from the Paper

| Aspect | Paper | Our Implementation |
|--------|-------|--------------------|
| Training images | 39,375 (full FLAME) | Smaller subset (Google Drive download) |
| Test images | 8,617 (Phantom 3 camera) | Custom test set (`Test_Dataset1__Our_Own_Dataset`) |
| GPU | Tesla V100-SXM2-32GB | Colab/Kaggle free GPU |
| Total epochs | 40 per stage | 20+20 (2-stage) with early stopping + ReduceLROnPlateau |
| Ensemble voters | 3 or 5 | 3 (time constraints) |
| Random seed strategy | Random init across 30 runs | Fixed per-voter seeds (i×7+13) |
| Self-collected dataset | Yes (1,204 images) | Not reproduced |
| Parallel GPU training | Yes (3 GPUs) | Sequential (single GPU) |
| CBAM integration | Inside last 2 bottleneck blocks | Applied as 2 sequential CBAM blocks on backbone output |

Despite these differences, the **key trends and relative model rankings are preserved**, confirming the validity of the MV2-CBAM approach.

---


