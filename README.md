# Multimodal Product Intelligence

A multimodal machine learning system for product price prediction that combines visual information from product images and textual information from product titles. The project leverages pretrained CLIP and DistilBERT embeddings and learns a fusion network to estimate product prices from Amazon product listings.


## Overview

Product prices are influenced by multiple factors, including product appearance, branding, quality indicators, and textual descriptions. Traditional approaches often rely on either image data or text data alone, limiting their ability to capture the complete product context.

This project explores multimodal learning by combining:

* **CLIP** for visual feature extraction
* **DistilBERT** for textual feature extraction
* **PyTorch Fusion MLP** for multimodal regression

The objective is to predict product prices using both product images and titles while evaluating the contribution of each modality through baseline comparisons.


## Dataset

The dataset used in this project is publicly available on Kaggle:

**Amazon Products Dataset (2023, 1.4M Products)**
https://www.kaggle.com/datasets/asaniczka/amazon-products-dataset-2023-1-4m-products

Due to GitHub file size limitations, the dataset is not included in this repository.

After downloading, place the file in:

```text
data/amazon_products.csv
```


### Data Processing

* Removed missing values
* Filtered invalid price ranges
* Created a balanced dataset across price segments
* Downloaded and validated product images
* Resized images to 224×224
* Generated multimodal embeddings

Final dataset contained approximately 2,000 validated product samples after image filtering.


## Architecture

### Visual Encoder

**CLIP ViT-B/32**

* Pretrained OpenAI CLIP model
* Generates 512-dimensional image embeddings
* Frozen during training

### Text Encoder

**DistilBERT**

* Pretrained transformer language model
* Generates 768-dimensional text embeddings
* Frozen during training

### Feature Fusion

Image and text embeddings are concatenated:

512 (CLIP) + 768 (DistilBERT) = 1280-dimensional feature vector

### Fusion Network

```text
Input (1280)
      │
      ▼
Linear (512)
      │
    ReLU
      │
 Dropout (0.3)
      │
      ▼
Linear (128)
      │
    ReLU
      │
 Dropout (0.2)
      │
      ▼
Linear (1)
      │
 Predicted Price
```


## Methodology

### Step 1: Data Preparation

* Cleaned Amazon product dataset
* Balanced price categories
* Downloaded product images in parallel
* Filtered invalid image URLs

### Step 2: Feature Extraction

Generated:

* CLIP image embeddings
* DistilBERT text embeddings

Combined both embeddings into a single multimodal representation.

### Step 3: Baseline Models

Three baseline models were trained:

1. Image-only (CLIP + Ridge Regression)
2. Text-only (DistilBERT + Ridge Regression)
3. Fusion (CLIP + DistilBERT + Ridge Regression)

### Step 4: Fusion MLP

A deep neural network was trained on fused embeddings to learn nonlinear interactions between visual and textual features.


## Results
```text

| Model                          | SMAPE      |
------------------------------------------------
| Image-only (CLIP + Ridge)      | 71.36%     |
| Text-only (DistilBERT + Ridge) | 69.03%     |
| Fusion (Ridge, no MLP)         | 78.50%     |
| Fusion MLP (CLIP + DistilBERT) | 45.45%     |
```
### Key Findings

* Fusion MLP achieved the best overall performance.
* Deep multimodal fusion significantly outperformed linear fusion.
* Combining image and text information improved predictive performance compared to single-modality models.
* The model achieved a **23.58% SMAPE improvement** over the strongest baseline.


## Tech Stack

### Machine Learning

* PyTorch
* Scikit-Learn
* NumPy
* Pandas

### Foundation Models

* OpenAI CLIP
* DistilBERT (Hugging Face Transformers)

### Data Processing

* PIL
* Requests
* Concurrent Futures


## Project Structure

```text
Multimodal-Product-Intelligence/
│
├── data/
│   ├── amazon_products.csv
│   ├── fused_emb.npy
│   └── prices.npy
│
├── notebooks/
│   └── Multimodal_Product_Intelligence.ipynb
│
├── models/
│   └── fusion_mlp.pt
│
└── README.md
```


## Future Improvements

* Train on larger product catalogs
* Fine-tune CLIP and DistilBERT encoders
* Experiment with attention-based fusion mechanisms
* Incorporate product metadata and category information
* Deploy as a real-time pricing intelligence application


## Conclusion

This project demonstrates the effectiveness of multimodal learning for product price prediction. By combining CLIP visual embeddings and DistilBERT textual embeddings through a neural fusion architecture, the model substantially outperformed image-only, text-only, and linear fusion baselines, highlighting the value of integrating complementary information from multiple modalities.
