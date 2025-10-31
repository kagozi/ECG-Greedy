# ECG-Greedy: Multi-Modal ECG Classification with CWT Scalograms & Phasograms 
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
<!-- [![Paper](https://img.shields.io/badge/ISBI-2025-blue)](https://biomedicalimaging.org/2025/)   -->
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)

> **SOTA ensemble AUC 0.9233** on **PTB-XL Super-Diagnostic Task** using **early fusion of scalograms + phasograms + raw signals**.

This repository implements the **multi-representation ECG classification framework** from our **ISBI 2025** paper, combining:
- **Time-domain**: XResNet1d101 (raw ECG)
- **Time-frequency**: CWT → **Scalograms**
- **Phase-domain**: CWT → **Phasograms**
- **Fusion**: Early & Late
- **Backbones**: Custom 2DCNN, ResNet50, EfficientNet-B0, Swin, **Hybrid Swin**

---

## Highlights
- **Ensemble AUC: 0.9233** (beats XResNet1d101: 0.9224)
- **Hybrid Swin** best single backbone (0.9018 early fusion)
- **Early > Late > Single modality**
- Full preprocessing, training, and evaluation pipeline
- Reproducible on **PTB-XL** with patient-wise splits 

> **Note:** All the commands are based on a Unix based system.
> For a different system look for similar commands for it.


## Setup

We are using Python version 3.11.9

```bash
$ python --version
Python 3.11.9
```
### Requirements

```bash
# Clone and install
git clone https://github.com/kagozi/MultiModal-ECG.git
cd MultiModal-ECG

pip install -r requirements.txt
```
### Python virtual environment

**Create** a virtual environment:

```bash
python3 -m venv .venv
```

`.venv` is the name of the folder that would contain the virtual environment.

**Activate** the virtual environment:

```bash
source .venv/bin/activate
```

**Windows**
```bash
source .venv/Scripts/activate
```
# Quick Start
## Download the PTBXL dataset to a preferred directory: 
```bash
wget https://physionet.org/content/ptb-xl/get-zip/1.0.3/
```

## Run full pipeline: preprocess → train → evaluate → ensemble

```bash
python3 load_and_standardize.py # creates standardized raw signal representations
python3 generate_cwt.py # creates Wavelet representations (Phasogram and Scalograms)
python3 benchmark.py # Trains XResNet1d101 on raw signals
python3 train_models.py # Trains CWT based models on Phasogram and Scalograms
python3 tests_ensemble.py # Run tests, generate confusion matrices, and evaluate different ensembling strategies
```




