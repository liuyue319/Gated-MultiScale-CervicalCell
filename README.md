## 📄 Notice
This code is directly related to a manuscript currently under review at **The Visual Computer**. 
If you find this work useful, please cite our paper (Gated Multi-Scale Feature Fusion for Robust Cervical Cell Classification in Medical Imaging).

markdown
# Gated Multi-Scale Feature Fusion for Robust Cervical Cell Classification in Medical Imaging

This repository provides the code for gated feature fusion classification on the **SipakMed** dataset. It uses **DenseNet** and **RepLKNet** as feature extractors, fuses their features through a gating mechanism, and evaluates the classification performance. All scripts are designed to run directly after setting up the environment and data.

## Directory Structure
.
├── fea_densenet.py # DenseNet feature extraction
├── fea_replknet.py # RepLKNet feature extraction
├── gate.py # Gated fusion training & evaluation
├── requirements.txt # Python dependencies
├── config/
│ └── all_experiments.yaml # Configuration for the gated fusion
├── data/
│ └── prepare_sipakmed.py # Dataset splitting script
├── features/ # (generated) Extracted features
│ ├── densenet/ # DenseNet features
│ └── replknet/ # RepLKNet features
├── results/ # Standalone single‑model results
│ ├── densenet_standalone/
│ └── replknet_standalone/
└── classicify2/
└── feature_fusion_5/ # Gated fusion results
├── checkpoints/ # Saved model weights
├── figures/ # Confusion matrix and loss curves
└── gated_feature_fusion_summary.json # Performance metrics

text

## Environment Setup

1. **Clone the repository** and navigate to the project root.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
Download the SipakMed dataset (cervical cell images, 5 classes) and place it in the data/ directory. The expected structure and any path adjustments can be made inside data/prepare_sipakmed.py or the configuration file.

Data Preparation
Run the data split script to generate training/validation/test partitions:

bash
python data/prepare_sipakmed.py
Check the script’s internal variables to ensure the dataset root path is correct.

Feature Extraction
Extract features using the two pretrained backbones. The results will be saved into the features/ folder.

bash
# DenseNet features
python fea_densenet.py

# RepLKNet features
python fea_replknet.py
If the features/ directory already contains pre‑extracted files, you may skip this step.

Gated Fusion Training & Evaluation
Once the features are ready, run the gated fusion experiment:

bash
python gate.py
This script will:

Load features from features/densenet/ and features/replknet/

Train the gated fusion classifier according to config/all_experiments.yaml

Save checkpoints, loss curves, confusion matrices, and a summary JSON into classicify2/feature_fusion_5/

Results
Standalone baselines: results/densenet_standalone/ and results/replknet_standalone/ contain the results of the two individual backbone classifiers.

Fusion model:

classicify2/feature_fusion_5/checkpoints/ – trained model weights.

classicify2/feature_fusion_5/figures/ – confusion matrix and training/validation loss plots.

classicify2/feature_fusion_5/gated_feature_fusion_summary.json – final accuracy, precision, recall, F1‑score, etc.

Configuration
All gated fusion hyperparameters, data paths, and training options are documented in config/all_experiments.yaml for reference only. To actually change:

Batch size, learning rate, number of epochs

Feature dimensions and fusion settings

please modify the corresponding hard-coded values in the source code .

Output directories

Requirements
Main packages (see requirements.txt for exact versions)

Citation

Moreover, you are encouraged to urge readers to cite this relevant manuscript.

Dataset Citation

- **SIPaKMeD**: M. E. Plissiti et al., "SIPAKMED: A new dataset for ...", ICIP 2018. DOI: 10.1109/ICIP.2018.8451788. URL: https://www.cs.uoi.gr/~marina/sipakmed.html

- **Mendeley Data**: Cite the specific dataset page (Author, Year, Title, DOI: 10.17632/xxxxx.x). URL: https://data.mendeley.com/

- **Herlev**: J. Jantzen et al., "Pap-smear benchmark data for pattern classification", NiSIS 2005. URL: https://opendatalab.org.cn/OpenDataLab/HErlev

