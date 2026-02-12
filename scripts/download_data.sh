#!/bin/bash

mkdir -p data
cd data
for i in {1..47}; do
     wget https://cernbox.cern.ch/remote.php/dav/public-files/bE1UoOQJry001ih/data/output_Phase2_HGCalL1T_Clustering_${i}_latent.npz
     wget https://cernbox.cern.ch/remote.php/dav/public-files/bE1UoOQJry001ih/data/output_Phase2_HGCalL1T_Clustering_${i}.root
done
cd -
mkdir -p encoders
cd encoders
for i in {2..5}; do
     wget https://cernbox.cern.ch/remote.php/dav/public-files/bE1UoOQJry001ih/encoders/encoder_model_NoBiasModel_elink_${i}.hdf5
done
cd -

wget https://cernbox.cern.ch/remote.php/dav/public-files/bE1UoOQJry001ih/encode_wafers.py
wget https://cernbox.cern.ch/remote.php/dav/public-files/bE1UoOQJry001ih/environment.yml
wget https://cernbox.cern.ch/remote.php/dav/public-files/bE1UoOQJry001ih/load_events.py
wget https://cernbox.cern.ch/remote.php/dav/public-files/bE1UoOQJry001ih/README.md

