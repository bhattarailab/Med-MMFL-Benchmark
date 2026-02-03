# Med-MMFL: A Multimodal Federated Learning Benchmark in Healthcare
This codebase provides a Pytorch implementation of:

> **Med-MMFL: A Multimodal Federated Learning Benchmark in Healthcare.**  
[![Med-MMFL](https://img.shields.io/badge/MMFL-Benchmark-4c72b0?style=for-the-badge)](https://github.com/bhattarailab/fmlbenchmark)  
Aavash Chhetri*, Bibek Niroula*, Pratik Shrestha, Yash Raj Shrestha, Lesley A. Anderson,  
Prashnna K. Gyawali, Loris Bazzani, Binod Bhattarai†  
\* Equal contribution  
† Corresponding author  


---

## Abstract

Federated learning (FL) enables collaborative model training across decentralized medical institutions while preserving data privacy. However, medical FL benchmarks remain scarce, with existing efforts focusing mainly on unimodal or bimodal modalities and a limited range of medical tasks. This gap underscores the need for standardized evaluation to advance systematic understanding in medical MultiModal FL (MMFL). To this end, we introduce Med-MMFL, the first comprehensive MMFL benchmark for the medical domain, encompassing diverse modalities, tasks, and federation  scenarios. Our benchmark evaluates six representative state-of-the-art FL algorithms, covering different aggregation strategies, loss formulations, and regularization techniques. It spans datasets with 2 to 4 modalities, comprising a total of 10 unique medical modalities, including text, pathology images, ECG, X-ray, radiology reports, and multiple MRI sequences. Experiments are conducted across naturally federated, synthetic IID, and synthetic non-IID settings to simulate real-world heterogeneity. We assess segmentation, classification, modality alignment (retrieval), and VQA tasks. To support reproducibility and fair comparison of future multimodal federated learning (MMFL) methods under realistic medical settings, we release the complete benchmark implementation, including data processing and partitioning pipelines, at [bhattarailab/Med-MMFL-Benchmark](https://github.com/bhattarailab/Med-MMFL-Benchmark).

---

![Overview of our proposed Med-MMFL benchmark framework. It spans diverse multimodal medical datasets, task types, and client partitioning strategies, integrating multiple FL algorithms to provide a unified evaluation platform.](./figures/Med-MMFL-central.png)

**Code Release Coming Soon**

The following components will be made available on this GitHub repository shortly:

- **Complete benchmark codebase**
- **Model implementations**
- **Federated learning algorithms**
- **Client-level data partitioning scripts**
  - Natural (real-world)
  - Synthetic IID
  - Synthetic non-IID
- **Training and evaluation scripts**
- **Reproducible experiment configurations**

Please stay tuned for updates.
