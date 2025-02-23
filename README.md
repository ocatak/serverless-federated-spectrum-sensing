# Serverless Federated Learning for Spectrum Sensing

This repository contains the code and resources for the paper titled **"Serverless Federated Learning for Spectrum Sensing"** published in [Journal Name](https://www.sciencedirect.com/science/article/abs/pii/S1874490725000370). The project focuses on implementing a decentralized federated learning framework for spectrum sensing tasks, leveraging a U-Net architecture for segmentation and classification of spectrograms.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Introduction
The goal of this project is to develop a decentralized federated learning (FL) framework for spectrum sensing, where multiple clients collaboratively train a model without sharing their local data. The framework is designed to handle non-IID (non-independent and identically distributed) data distributions, which are common in real-world scenarios. The model used is a U-Net architecture, which is well-suited for segmentation tasks in spectrogram data.

## Dataset
The dataset used in this project is available at [RadarSpectrumSensing-FL-AML](https://github.com/ocatak/RadarSpectrumSensing-FL-AML). It consists of spectrogram data and corresponding labels for different signal types, including Noise, LTE, NR, and Radar.

### Dataset Structure
- **Spectrograms**: Mat files containing the spectrogram data.
- **Labels**: CSV files containing the pixel-wise labels for each spectrogram.

## Installation
To set up the environment and run the code, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ocatak/serverless-federated-learning-spectrum-sensing.git
   cd serverless-federated-learning-spectrum-sensing
   ```

2. **Install dependencies**:
   Ensure you have Python 3.8 or later installed. Then, install the required packages using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   The dataset will be automatically downloaded and unzipped when you run the script for the first time. Alternatively, you can manually download it from the [dataset repository](https://github.com/ocatak/RadarSpectrumSensing-FL-AML).

## Usage
The main script for running the federated learning process is `serverless-federated-learning-spectrum-sensing.py`. Below are the steps to run the code:

1. **Run the centralized training**:
   To train a baseline model on the entire dataset without federated learning, run:
   ```bash
   python serverless-federated-learning-spectrum-sensing.py --mode centralized
   ```

2. **Run decentralized federated learning**:
   To run the decentralized federated learning process, use:
   ```bash
   python serverless-federated-learning-spectrum-sensing.py --mode decentralized
   ```

3. **Evaluate the model**:
   After training, you can evaluate the model on the validation set using:
   ```bash
   python serverless-federated-learning-spectrum-sensing.py --mode evaluate
   ```

### Command Line Arguments
- `--mode`: Specify the mode of operation (`centralized`, `decentralized`, or `evaluate`).
- `--num_clients`: Number of clients in the federated learning setup (default: 5).
- `--num_rounds`: Number of federated learning rounds (default: 10).
- `--epochs`: Number of epochs for local training (default: 3).

## Results
The results of the experiments, including model accuracy, divergence, and communication metrics, are saved in the `results/` directory. You can visualize the results using the provided plotting scripts.

### Sample Results
- **Model Divergence**: The divergence between models across different clients over federated learning rounds.
- **Accuracy**: Training and validation accuracy for both centralized and decentralized models.
- **Segmentation Metrics**: Dice coefficient, IoU, precision, recall, and F1 score for each class.

## Citation
If you use this code or the results from this project in your research, please cite the following paper:

```bibtex
@article{CATAK2025102634,
title = {Serverless federated learning: Decentralized spectrum sensing in heterogeneous networks},
journal = {Physical Communication},
pages = {102634},
year = {2025},
issn = {1874-4907},
doi = {https://doi.org/10.1016/j.phycom.2025.102634},
url = {https://www.sciencedirect.com/science/article/pii/S1874490725000370},
author = {Ferhat Ozgur Catak and Murat Kuzlu and Yaser Dalveren and Gokcen Ozdemir},
keywords = {Federated learning (FL), Decentralized FL, Non-IID, Spectrum sensing},
abstract = {Federated learning (FL) has gained more popularity due to the increasing demand for robust and efficient mechanisms to ensure data privacy and security during collaborative model training in the concept of artificial intelligence/machine learning (AI/ML). This study proposes an advanced version of FL without the central server, called a serverless or decentralized federated learning framework, to address the challenge of cooperative spectrum sensing in non-independent and identically distributed (non-IID) environments. The framework leverages local model aggregation at neighboring nodes to improve robustness, privacy, and generalizability. The system incorporates weighted aggregation based on distributional similarity between local datasets using Wasserstein distance. The results demonstrate that the proposed serverless federated learning framework offers a satisfactory performance in terms of accuracy and resilience.}
}
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
- You can also include a `requirements.txt` file listing all the Python dependencies needed to run the code.

Let me know if you need further customization!
