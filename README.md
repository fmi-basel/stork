# Stork

Stork is a library for training spiking neural networks (SNNs) with backpropagation through time (BPTT) using Pytorch.

Copyright 2019-2021 Julian Rossbroich, Julia Gygax and Friedemann Zenke

## Citing Stork

If you find this library useful and use it for your research projects, please cite

Rossbroich, J., Gygax, J. & Zenke, F. Fluctuation-driven initialization for spiking neural network training


# Setup

Create and activate a virtual environment.

If you use conda, run the following commands to create and activate the conda environment (use python version >= 3.6).

```bash
conda create --name stork_venv python=3.6 pip
conda activate stork_venv
pip install -r requirements.txt
```

If you additionally want to install stork, run
```bash
pip install -e .
```
in this directory, what will run `setup.py` for stork.

# Examples

The `examples` directory contains Notebooks and Python Scripts that contain examples of different complexities. 


![Stork Logo](img/stork_logo_small.png)
Logo copyright 2019 Koshika Yadava


