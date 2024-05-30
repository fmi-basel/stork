import gzip
import pickle
import torch
import string
import random
import time
import numpy as np


def get_random_string(string_length=5):
    """ Generates a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_length))


def get_basepath(dir=".", prefix="default", salt_length=5):
    """ Returns pre-formatted and time stamped basepath given a base directory and file prefix. """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if salt_length:
        salt = get_random_string(salt_length)
        basepath = "%s/%s-%s-%s" % (dir, prefix, timestr, salt)
    else:
        basepath = "%s/%s-%s" % (dir, prefix, timestr)
    return basepath


def write_to_file(data, filename):
    """Writes an object/dataset to zipped pickle.

    Args:
        data: the (data) object
        filename (str): the filename to write to
    """
    fp = gzip.open("%s" % filename, 'wb')
    pickle.dump(data, fp)
    fp.close()


def load_from_file(filename):
    """ Loads an object/dataset from a zipped pickle. """
    fp = gzip.open("%s" % filename, 'r')
    data = pickle.load(fp)
    fp.close()
    return data


def to_sparse(x):
    """ converts dense tensor x to sparse format """

    indices = torch.nonzero(x)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return torch.sparse.FloatTensor(indices, values, x.size(), device=x.device)


def get_lif_kernel(tau_mem=20e-3, tau_syn=10e-3, dt=1e-3):
    """ Computes the linear filter kernel of a simple LIF neuron with exponential current-based synapses.

    Args:
        tau_mem: The membrane time constant
        tau_syn: The synaptic time constant
        dt: The timestep size

    Returns:
        Array of length 10x of the longest time constant containing the filter kernel

    """
    tau_max = np.max((tau_mem, tau_syn))
    ts = np.arange(0, int(tau_max*10/dt))*dt
    n = len(ts)
    kernel = np.empty(n)
    I = 1.0  # Initialize current variable for single spike input
    U = 0.0
    dcy1 = np.exp(-dt/tau_mem)
    dcy2 = np.exp(-dt/tau_syn)
    for i, t in enumerate(ts):
        kernel[i] = U
        U = dcy1*U + (1.0-dcy1)*I
        I *= dcy2
    return kernel


def lif_membrane_dynamics(input_current, tau_mem, dt=1e-3):
    """
    Computes LIF membrane dynamics for n postsynaptic neurons.
    Input_current is a torch.tensor of shape [ts] or [n, ts]
    
    returns a kernel of shape [ts] or [n, ts]
    """
    
    if input_current.dim() == 1:
        U = 0.0
        n = 1
        ts = np.arange(len(input_current)) * dt
    else:
        U = torch.zeros(input_current.shape[0])  
        n = input_current.shape[0]  
        ts = np.arange(input_current.shape[1]) * dt
        
    kernel = torch.empty(input_current.shape)
    dcy = np.exp(-dt/tau_mem)
    for i, t in enumerate(ts):
        kernel[..., i] = U
        U = dcy * U + (1.0-dcy) * input_current[..., i]
        
    return kernel


def convlayer_size(nb_inputs, kernel_size, padding, stride):
    """
    Calculates output size of convolutional layer
    """
    res = ((np.array(nb_inputs) - kernel_size + 2 * padding) / stride) + 1
    return res
