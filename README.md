Multimodal Audio-Lyrics Clustering with Beta-VAE

 📌 Project Overview
This project implements a Deep Learning approach to cluster music tracks by fusing Audio Spectrograms and Lyrics. 



 📂 Project Structure
The file structure follows the project requirements:

```text
CSE425-Audio-Lyrics-Clustering/
├── data/                       # Dataset (Audio .mp3 and Lyrics .csv)
├── notebooks/
│   ├── Easy_Task.ipynb         # Baseline Linear VAE (Audio Only)
│   ├── Medium_Task.ipynb       # ConvVAE + Hybrid Clustering
│   └── Hard_Task.ipynb         # Beta-VAE + Advanced Metrics (Main Entry Point)
├── results/
│   ├── latent_visualization/   # Generated plots (Latent space, Purity, Reconstruction)
│   ├── clustering_metrics.csv  # Final comparison table
│   └── beta_vae_model.pth      # Saved model weights
├── src/
│   ├── vae.py                  # PyTorch Model Definitions (ConvVAE, Beta-VAE)
│   ├── dataset.py              # Custom PyTorch Dataset Loaders
│   ├── clustering.py           # Clustering Logic (KMeans, DBSCAN)
│   └── evaluation.py           # Metrics (Purity, NMI, ARI)
└── README.md