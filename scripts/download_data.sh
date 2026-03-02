#!/bin/bash

CERNBOX_LINK=$1

mkdir -p data
cd data
for i in {1..47}; do
     wget $CERNBOX_LINK/data/output_Phase2_HGCalL1T_Clustering_${i}_latent.npz
     wget $CERNBOX_LINK/data/output_Phase2_HGCalL1T_Clustering_${i}.root
done
cd -
mkdir -p encoders
cd encoders
for i in {2..5}; do
     wget $CERNBOX_LINK/encoders/encoder_model_NoBiasModel_elink_${i}.hdf5
done
cd -

wget $CERNBOX_LINK/encode_wafers.py
wget $CERNBOX_LINK/environment.yml
wget $CERNBOX_LINK/load_events.py
wget $CERNBOX_LINK/README.md

