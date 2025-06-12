# AI-Based Quality Control for Stamping Machines

An AI-powered multimodal quality control system for industrial stamping machines. This project combines real-time image inspection and force signal analysis using deep learning (ResNet, LSTM, VAE) and traditional machine learning techniques (Random Forest, SVM, PCA). Developed in Python with PyTorch and Scikit-learn, this system aims to detect surface flaws, structural deformations, and predict anomalies to prevent equipment failure.

🔗 [Project Webpage](https://liuyumaosid.wixsite.com/resume-of-yumao-liu/ai4stamping-project)

---
## 🧪 Industrial Background

This research was conducted as part of a collaboration with **PtU – Institute for Production Engineering and Forming Machines, TU Darmstadt**, focusing on quality control in the production of **deep drawn paper cups**.

<div align="center">
  <img src="assets/ptu.avif" width="300"/>
  <p><em>Stamping machine of PtU.</em></p>
</div>

---

## 🎯 Task Definition

- Install vision and force sensors to monitor the production process.
- Collect and process sensory data from:
  - **Side and top-view cameras**
  - **Five-dimensional time-series**: `time`, `displacement`, `velocity`, `acceleration`, and `force`
- Develop ML-based quality control methods to:
  - Evaluate stamping force curves
  - Detect defects and predict failures
  - Benchmark image and signal processing techniques

---

## 🔍 Key Features

- **Visual Inspection (ResNet CNN)**: Detects deformations and surface flaws with high accuracy using a ResNet-based classifier.
- **Time-Series Anomaly Detection (LSTM)**: Identifies anomalies in stamping force signal using an Encoder-Decoder architecture.
- **Joint Anomaly Detection (VAE)**: Reconstructs both visual and signal data to detect unknown or noisy defects.
- **Traditional ML Baseline**: Random Forest, SVM, and PCA were explored to benchmark signal-based anomaly detection.

---


## 🖼️ Sample Visualizations

### 🔹 Clustering of Visual Samples using ResNet + PCA

| Sound Paper Cups | Broken Paper Cups |
|------------------|-------------------|
| ![Clustered Sound Cups](./assets/clustered_sound.png) | ![Clustered Broken Cups](./assets/clustered_broken.png) |

These images illustrate the clustering outcome of side-view images using ResNet features reduced by PCA. The clustering helps visually differentiate sound and defective samples.

---

### 🔹 Force Curve Reconstruction using Conventional VAE

| Sound Drawing Process | Broken Drawing Process |
|------------------------|------------------------|
| ![Low Loss Curve](./assets/vae_recon_good.png) | ![High Loss Curve](./assets/vae_recon_bad.png) |

The VAE learns to reconstruct the full force signal of each stamping cycle. For sound samples, the reconstruction error is low; for broken ones, the mismatch is clear and significant.

---

### 🔹 Fine-Grained Anomaly Detection using LSTM Autoencoder

| LSTM Architecture Overview | Detected Anomalies |
|----------------------------|--------------------|
| ![LSTM Architecture](./assets/lstm_structure.png) | ![Pointwise Anomalies](./assets/lstm_anomaly_output.png) |

The LSTM-Autoencoder enables continuous, time-point-level anomaly detection by learning long-term dependencies in stamping force signals. Red points indicate detected anomalies.

---

## 🗂️ Repository Structure

```
├── GIT_ResNet_KMeans_top.ipynb              # Image clustering + ResNet
├── GIT_ResNet_KMeans_side.ipynb             # Side-view defect detection
├── GIT_newtrain_convolutional_VAE.ipynb     # VAE training and reconstruction
├── GIT_Force_Time_Series_Anomaly_Detection_using_AutoEncoders_(LSTM)_in_Pytorch.ipynb
├── GIT_Time_Series_of_Price_Anomaly_Detection_with_LSTM_Autoencoders_(Keras).ipynb
```

---

## 📈 Results Summary

| Module         | Task                          | Metric              |
|----------------|-------------------------------|---------------------|
| ResNet         | Image classification          | Accuracy: 94%       |
| LSTM Autoencoder | Force signal forecasting     | AUC: 0.91           |
| VAE            | Joint anomaly detection        | Recon. Error ≤ 0.05 |

---

## 🚀 Setup Instructions

1. Clone this repository
    ```bash
    git clone https://github.com/yourusername/stamping-quality-control.git
    cd stamping-quality-control
    ```
2. Open desired `.ipynb` file in Jupyter or Google Colab.
3. Upload your dataset or use demo data within notebooks.
4. Run cell-by-cell for model training, evaluation, and visualization.

---

## 📌 Novel Contributions

- First system integrating **joint visual + signal anomaly detection** with deep architectures in a stamping context.
- Robust **convolutional VAE** supports noisy, real-world datasets.
- Achieves **real-time inference** and defect classification with high reliability.
- Traditional ML methods benchmarked and compared for interpretability and efficiency.

---

## 📬 Contact

For questions, collaboration, or access to demo videos, visit:
**[📄 Project Page](https://liuyumaosid.wixsite.com/resume-of-yumao-liu/copy-of-plcmproject)**  
Author: **Yumao Liu** | Email: `Liuyumao_SiD@outlook.com`

---
© Yumao Liu, 2025
