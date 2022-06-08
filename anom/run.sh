#!/usr/bin/env sh

# Install anomalib in devel mode.
cd /opt/project/anom/anomalib || exit
python -m pip install -e .

# Run GANomaly experiments.
for c in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper;
  do
    sed -i "s/category.*#/category: ${c}  #/" /opt/project/anom/anomalib/anomalib/models/ganomaly/config.yaml
    python anom/ganomaly.py
  done
