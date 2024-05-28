<img align="right" width=30% src="img/stork_logo_cr.png">

# Stork

Stork is a library designed for the training of spiking neural networks (SNNs). In contrast to conventional deep learning methods, SNNs operate on spikes instead of continuous activation functions, this is why stork extends PyTorch's auto-differentiation capabilities with surrogate gradients (<a href="https://direct.mit.edu/neco/article-abstract/30/6/1514/8378/SuperSpike-Supervised-Learning-in-Multilayer?redirectedFrom=fulltext">Zenke & Ganguli, 2018</a>) to enable the training of SNNs with backpropagation through time (BPTT).  
Stork supports leaky integrate-and-fire (LIF) neurons including adaptive LIF neurons and different kinds of synaptic connections allowing to use e.g. Dalian and Convolutional layers as well as constructing network architectures with recurrent or skip connections. For each neuron group, customizable activity regularizers are available to e.g. apply homeostatic plasticity.  
Furthermore, stork uses per default initialization in the fluctuation-driven regime, what enhances SNN training especially in deep networks.


## Citing Stork

If you find this library useful and use it for your research projects, please cite

<a href="https://iopscience.iop.org/article/10.1088/2634-4386/ac97bb">Rossbroich, J., Gygax, J., and Zenke, F. (2022).  
Fluctuation-driven initialization for spiking neural network training.  
Neuromorph. Comput. Eng.  </a>

**Bibtex Citation:**

```
@article{rossbroich_fluctuation-driven_2022,
 title = {Fluctuation-driven initialization for spiking neural network training},
 author = {Rossbroich, Julian and Gygax, Julia and Zenke, Friedemann},
 doi = {10.1088/2634-4386/ac97bb},
 journal = {Neuromorphic Computing and Engineering},
 year = {2022},
}
```


# Setup

1. Create and activate a virtual environment.
2. Download stork or clone the repository with
	```bash
	git clone <git@github.com>:fmi-basel/stork.git
 	```
3. Change into the stork directory.
	```bash
 	cd stork
 	```
4. Install the requirements with
	```bash
 	pip install -r requirements.txt
 	```
5. Install `stork` with
	```bash
 	pip install -e .
 	```


# Examples

The `examples` directory contains notebooks and Python scripts that contain examples of different complexities.
- **[00_FluctuationDrivenInitialization](examples/00_FluctuationDrivenInitialization.ipynb):** This example will provide some intuition about the idea behind fluctuation-driven initialization and will reproduce the panels from Fig. 1
- **[01_Shallow_SNN_Randman](examples/01_Shallow_SNN_Randman.ipynb):** This demonstrates how to train an SNN with a single hidden layer on the <a href="https://github.com/fzenke/randman">RANDom MANifold (Randman)</a> dataset.
- **[02_Deep_SNN_SHD](examples/02_Deep_SNN_SHD.ipynb):** This example demonstrates how to train a deep feedforward SNN with multiple hidden layers on the <a href="https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/">Spiking Heidelberg Digits (SHD)</a> dataset.
- **[03_Deep_ConvSNN_SHD](examples/03_Deep_ConvSNN_SHD.ipynb):** Here we provide an example of a deep recurrent convolutional SNN on the SHD dataset. This example will introduce the use of [layer](stork/layers.py) to create convolutional layers.
- **[04_DalesLaw_SNN_SHD](examples/04_DalesLaw_SNN_SHD.ipynb):** This notebook demonstrates how to implement a Dalian network, meaning networks with separate populations of excitatory and inhibitory neurons (i.e. the synaptic connections are sign constrained), by using the `DalianLayer` class from the [layer](stork/layers.py) module.
- **[05_Deep_ConvSNN_DVS-Gestures](examples/05_Deep_ConvSNN_DVS-Gestures.ipynb):** Similar to **[03_Deep_ConvSNN_SHD](examples/03_Deep_ConvSNN_SHD.ipynb):**, but for the [DVS128 Gesture](https://research.ibm.com/interactive/dvsgesture/) dataset.


## Funding
The development of Stork was supported by the Swiss National Science Foundation [grant number PCEFP3_202981].
