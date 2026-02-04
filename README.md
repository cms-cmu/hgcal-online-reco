# HGCAL Online Reconstruction

Repository for HGCAL online reconstruction using machine learning techniques. This repository contains code and tools for training and evaluating machine learning models on HGCAL simulated datasets.

## Light-weight transformers

Here is reported important documentation and links of the project aiming at the implementation of light-weight transformers and their deployment on FPGAs at the L1 trigger.

Important links:
- Detector clustering transformers and GNNs: [Google slides](https://drive.google.com/file/d/1ebNVpTzbxWo1XX1tmrEB542-bl2xnC9L/view)
- Chenrui's repository with transformer-based models evaluated on synthetic detector data: [GitHub repo](https://github.com/CrystarRay/Evaluating_Transformer_Based_Models_for_Clustering_in_Synthetic_Detector_Data)
- Datasets including the latent space vectors for the HGCAL wafers produced by Peter: [CERNBox](https://cernbox.cern.ch/s/bE1UoOQJry001ih)
- Description of the conditional autoencoder of which we are analyzing the encoded latent space: [CAE Docs](https://cmucms-online-ml.docs.cern.ch/hgcal/cae/)

### Open ROOT files in Python

Here are some useful reference links to open and analyze ROOT files in python or with the ROOT software:

- Uproot docs: https://uproot.readthedocs.io/en/latest/
- Awkward arrays docs: https://awkward-array.org/doc/main/
- If you want to install ROOT on your laptop, here is the installation guide (but it's not necessary, you can just use uproot to open ROOT files in Python): https://root.cern/install/

