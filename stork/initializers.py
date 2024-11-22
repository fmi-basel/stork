# Initializers module
# J Rossbroich, 2021-22

import math
import torch
import torch.distributions as dists
import numpy as np

from . import connections
from . import layers
from . import nodes

from .utils import get_lif_kernel

# FUNCTIONS
# # # # # # # # # #

def _get_epsilon(calc_mode, tau_mem, tau_syn=None, timestep=1e-3):
    if calc_mode == 'analytical':
        return _epsilon_analytical(tau_mem, tau_syn)

    elif calc_mode == 'numerical':
        return _epsilon_numerical(tau_mem, tau_syn, timestep)

    else:
        raise ValueError('invalid calc mode for epsilon')

def _epsilon_analytical(tau_mem, tau_syn=None):
    if tau_syn is None:
        # using delta synapses
        epsilon_bar = tau_mem
        epsilon_hat = tau_mem / 2

        return epsilon_bar, epsilon_hat

    epsilon_bar = tau_syn
    epsilon_hat = (tau_syn ** 2) / (2 * (tau_syn + tau_mem))

    return epsilon_bar, epsilon_hat

def _epsilon_numerical(tau_mem, tau_syn, timestep):
    kernel = get_lif_kernel(tau_mem, tau_syn, timestep)
    epsilon_bar = kernel.sum() * timestep
    epsilon_hat = (kernel**2).sum() * timestep

    return epsilon_bar, epsilon_hat

# CLASSES
# # # # # # # # # #

class Initializer():
    """
    Abstract Base Class for Initializer Objects.
    """

    def __init__(self, scaling='1/sqrt(k)', sparseness=1.0, bias_scale=1.0, bias_mean=0.0):

        self.scaling = scaling
        self.sparseness = sparseness
        self.bias_scale = bias_scale
        self.bias_mean = bias_mean

    def initialize(self, target):

        if isinstance(target, connections.BaseConnection):
            self.initialize_connection(target)

        elif isinstance(target, layers.AbstractLayer):
            self.initialize_layer(target)

        else:
            raise TypeError(
                "Target object is unsupported. must be Connection or Layer instance.")

    def initialize_layer(self, layer):
        """
        Initializes all connections in a `Layer` object
        """

        for connection in layer.connections:
            self.initialize(connection)

    def initialize_connection(self, connection):
        """
        Initializes weights of a `Connection` object
        """
        raise NotImplementedError

    def _apply_scaling(self, weights, connection):
        """
        Implements weight scaling options
        """
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
            connection.op.weight)

        if self.scaling is None:
            return weights

        elif self.scaling == "1/sqrt(k)":

            return weights / math.sqrt(fan_in)

        elif self.scaling == "1/k":
            return weights / fan_in

        elif isinstance(self.scaling, float):
            return weights * self.scaling

        else:
            return weights

    def _apply_sparseness(self, weights):
        """
        Applies sparse mask to weight matrix
        """
        if self.sparseness < 1.0:
            x = torch.rand_like(weights)
            mask = x.le(self.sparseness)

            # apply mask & correct weights for sparseness
            weights.mul_(1.0 / math.sqrt(self.sparseness) * mask)

        return weights

    def _set_weights_and_bias(self, connection, weights=None):
        """
        Set weights and biases of a connection object
        """
        # set weights
        self._set_weights(connection, weights)

        # set biases if used
        self._set_biases(connection)

        # apply constraints
        connection.apply_constraints()

    def _set_weights(self, connection, weights):
        """
        Sets weights at connection object and applies scaling and sparseness
        """
        # apply scaling
        weights = self._apply_scaling(weights, connection)

        # apply sparseness
        weights = self._apply_sparseness(weights)

        # set weights
        with torch.no_grad():
            connection.op.weight.data = weights

    def _set_biases(self, connection):
        """
        Biases are always initialized from a uniform distribution and scaled by 1/sqrt(k)
        """
        if connection.op.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                connection.op.weight)
            bound = self.bias_scale / np.sqrt(fan_in)

            with torch.no_grad():
                connection.op.bias.uniform_(-bound+self.bias_mean,
                                      bound+self.bias_mean)

    def _get_weights(self, *params):
        raise NotImplementedError


class DistInitializer(Initializer):
    """
    Initializes synaptic weights by drawing from an arbitrary parameterized torch.distributions object.
    """

    def __init__(self,
                 dist=dists.Normal(0, 1),
                 **kwargs):

        super().__init__(**kwargs)

        # assert validity of distribution object
        assert isinstance(
            dist, dists.Distribution), 'Invalid distribution object. Must inherit from torch.distributions.Distribution'
        self.dist = dist

    def _get_weights(self, connection):
        shape = connection.op.weight.shape
        return self.dist.sample(shape)

    def initialize_connection(self, connection):

        # Sample weights
        weights = self._get_weights(connection)

        # Set weights
        self._set_weights_and_bias(connection, weights)


class KaimingNormalInitializer(Initializer):

    def __init__(self,
                 gain = 1.0,
                 **kwargs):

        super().__init__(scaling=None,  # Fixed to None, as scaling is implemented in the weight sampling
                         **kwargs)

        self.gain = gain

    def _get_weights(self, connection):

        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(connection.op.weight)

        # get distribution
        sigma = self.gain / np.sqrt(fan_in)
        dist = dists.Normal(0, sigma)

        # get weight matrix
        shape = connection.op.weight.shape
        weights = dist.sample(shape)

        return weights

    def initialize_connection(self, connection):

        # Sample weights
        weights = self._get_weights(connection)

        # Set weights
        self._set_weights_and_bias(connection, weights)


class FluctuationDrivenNormalInitializer(Initializer):
    """
    Implements flucutation-driven initialization as described in:
    Rossbroich, J., Gygax, J. & Zenke, F. Fluctuation-driven initialization for spiking neural network training. arXiv [cs.NE] (2022)

    params:

        Initialization parameters
        mu_U:      The target membrane potential mean
        xi:        The target distance between mean membrane potential and firing threshold in units of standard deviation

        Network and input parameters
        nu:         Estimated presynaptic firing rate (currently global for the whole network)
        timestep:   Simulation timestep (needed for calculation of epsilon numerically)

    The initialization method can additionally be parameterized through

    epsilon_calc_mode:          [set to `numerical` or `analytical`]
                                Analytical or numerical calculation of epsilon kernels (numerically is usually
                                better for large timesteps >=1ms)

    alpha:                      [set to float between 0 and 1]
                                Scales weights so that this proportion of membrane potential fluctuations
                                (variance) is accounted for by feed-forward connections.
    """

    def __init__(self,
                 mu_u,
                 xi,
                 nu,
                 timestep,
                 epsilon_calc_mode='numerical',
                 alpha=0.9,
                 **kwargs):

        super().__init__(scaling=None,  # None, as scaling is implemented in the weight sampling
                         **kwargs)

        self.mu_u = mu_u
        self.xi = xi
        self.nu = nu
        self.timestep = timestep
        self.epsilon_calc_mode = epsilon_calc_mode
        self.alpha = alpha

    def _calc_epsilon(self, dst):
        """
        Calculates epsilon_bar and epsilon_hat, the integrals of the PSP kernel from a target
        neuron group `dst`
        """
        tau_mem = dst.tau_mem
        if isinstance(dst, nodes.DeltaSynapseLIFGroup):
            tau_syn = None
            self.epsilon_calc_mode = 'analytical'
        else:
            tau_syn = dst.tau_syn
        ebar, ehat = _get_epsilon(self.epsilon_calc_mode,
                                  tau_mem,
                                  tau_syn,
                                  self.timestep)

        return ebar, ehat

    def _get_weights(self, connection, mu_w, sigma_w):

        shape = connection.op.weight.shape

        # sample weights
        weights = dists.Normal(mu_w, sigma_w).sample(shape)

        return weights

    def _get_weight_parameters_con(self, connection):
        """
        Calculates weight parameters for a single connection
        """

        theta = 1.0  # Theta (firing threshold) is hardcoded as in the LIFGroup class

        # Read out relevant attributes from connection object
        n, _ = torch.nn.init._calculate_fan_in_and_fan_out(connection.op.weight)

        if isinstance(connection.dst, nodes.FilterLIFGroup):
            ebar, ehat = connection.dst.get_epsilon_numerical(self.timestep)
        else:
            ebar, ehat = self._calc_epsilon(connection.dst)

        mu_w = self.mu_u / (n * self.nu * ebar)
        sigma_w = math.sqrt(1 / (n * self.nu * ehat) *
                           ((theta - self.mu_u) / self.xi) ** 2 - mu_w ** 2)

        return mu_w, sigma_w

    def initialize_connection(self, connection):
        """
        Initializes weights of a single `Connection` object
        """
        # get parameters
        mu_w, sigma_w = self._get_weight_parameters_con(connection)
        # get weights
        weights = self._get_weights(connection, mu_w, sigma_w)
        # set weights
        self._set_weights_and_bias(connection, weights)

    def _get_weight_parameters_dst(self, dst):
        """
        Calculates weight parameters for all connections targeting a
        neuron group `dst`
        """

        theta = 1.0  # Theta (firing threshold) is hardcoded as in the LIFGroup code

        ebar, ehat = self._calc_epsilon(dst)

        # Read out some properties of the afferent connections
        nb_recurrent = len([c for c in dst.afferents if c.is_recurrent])
        nb_ff = len(dst.afferents) - nb_recurrent

        if nb_recurrent >= 1:
            # If there is at least one recurrent connection, use alpha to scale the
            # contribution to the membrane potential fluctuations
            alpha = self.alpha
        else:
            # Otherwise alpha equals one (all contribution to fluctuations spread across feed-forward connections)
            alpha = 1.0

        # Sum of all inputs
        N_total = int(sum([torch.nn.init._calculate_fan_in_and_fan_out(c.op.weight)[0]
                        for c in dst.afferents]))

        # List with weight parameters for each connection
        params = []

        for c in dst.afferents:

            # Number of presynaptic neurons
            N, _ = torch.nn.init._calculate_fan_in_and_fan_out(c.op.weight)

            # mu_w is the same for all connections
            mu_w = self.mu_u / (N_total * self.nu * ebar)

            # Compute sigma scaling factor
            if c.is_recurrent:
                scale = (1 - alpha) / nb_recurrent
            else:
                scale = alpha / nb_ff

            # sigma_w for this connection
            sigma_w = math.sqrt(scale / (N * self.nu * ehat) *
                                ((theta - self.mu_u) / self.xi) ** 2 - mu_w ** 2)

            # append to parameter list
            params.append((mu_w, sigma_w))

        return params

    def initialize_layer(self, layer):

        # Loop through each population in this layer
        for neurons in layer.neurons:

            # Consider all afferents to this population
            # and compute weight parameters for each connection
            weight_params = self._get_weight_parameters_dst(neurons)

            # Initialize each connection
            for idx, connection in enumerate(neurons.afferents):

                # Read out parameters for weight distribution
                mu_w, sigma_w = weight_params[idx]
                # sample weights
                weights = self._get_weights(connection, mu_w, sigma_w)
                # set weights
                self._set_weights_and_bias(connection, weights)


class FluctuationDrivenCenteredNormalInitializer(FluctuationDrivenNormalInitializer):
    """
    Simpler version of the FluctuationDrivenNormalInitializer class.
    Here, the normal distribution is centered, so that initialization of synaptic weights can be
    achieved by setting a target membrane potential standard deviation sigma_u = 1/xi
    """

    def __init__(self,
                 sigma_u,
                 nu,
                 timestep,
                 epsilon_calc_mode='numerical',
                 alpha=0.9,
                 **kwargs):

        super().__init__(mu_u = 0.0,
                         xi = 1 / sigma_u,
                         nu=nu,
                         timestep = timestep,
                         epsilon_calc_mode = epsilon_calc_mode,
                         alpha = alpha,
                         **kwargs)


class FluctuationDrivenExponentialInitializer(FluctuationDrivenNormalInitializer):

    def __init__(self,
                 sigma_u,
                 nu,
                 timestep,
                 epsilon_calc_mode='numerical',
                 alpha=0.9,
                 **kwargs):

        super().__init__(mu_u = 0.0,                # Fixed to balanced state
                         xi = 1 / sigma_u,
                         nu = nu,
                         timestep = timestep,
                         epsilon_calc_mode = epsilon_calc_mode,
                         alpha = alpha,
                         **kwargs)

    def _calc_epsilon(self, dst):
        """
        Calculates epsilon_bar and epsilon_hat, the integrals of the PSP kernel from a target
        neuron group `dst`
        """
        ebar_exc, ehat_exc = _get_epsilon(self.epsilon_calc_mode,
                                  dst.tau_mem,
                                  dst.tau_exc,
                                  self.timestep)

        ebar_inh, ehat_inh = _get_epsilon(self.epsilon_calc_mode,
                                  dst.tau_mem,
                                  dst.tau_inh,
                                  self.timestep)

        return ebar_exc, ehat_exc, ebar_inh, ehat_inh

    def _get_weights(self, connection, rate):
        shape = connection.op.weight.shape
        # sample weights
        weights = dists.Exponential(rate).sample(shape)
        return weights

    def initialize_connection(self, connection):
        """
        Initializes weights of a single `Connection` object
        """
        raise NotImplementedError

    def _get_weight_parameters_dst(self, dst):
        """
        Calculates weight parameters for all connections targeting a
        neuron group `dst`
        """

        theta = 1.0  # Theta (firing threshold) is hardcoded as in the LIFGroup code
        ebar_exc, ehat_exc, ebar_inh, ehat_inh = self._calc_epsilon(dst)

        # Read out some properties of the afferent connections
        inh_cons = [c for c in dst.afferents if c.is_inhibitory]
        exc_cons = [c for c in dst.afferents if not c.is_inhibitory]
        exc_ff_cons = [c for c in exc_cons if not c.is_recurrent]
        exc_rec_cons = [c for c in exc_cons if c.is_recurrent]

        nb_inh = len(inh_cons)
        nb_exc = len(exc_cons)
        nb_exc_ff = len(exc_ff_cons)
        nb_exc_rec = len(exc_rec_cons)

        # Assert that there is at least one excitatory and one inhibitory connection
        assert nb_inh >=1, 'each neuron group must have at least one inhibitory connection'
        assert nb_exc >=1, 'each neuron group must have at least one excitatory connection'

        # Sum of all inputs
        N_total_exc = int(sum([torch.nn.init._calculate_fan_in_and_fan_out(c.op.weight)[0]
                               for c in exc_cons]))
        N_total_exc_rec = int(sum([torch.nn.init._calculate_fan_in_and_fan_out(c.op.weight)[0]
                                   for c in exc_rec_cons]))
        N_total_exc_ff = int(sum([torch.nn.init._calculate_fan_in_and_fan_out(c.op.weight)[0]
                                  for c in exc_ff_cons]))
        N_total_inh = int(sum([torch.nn.init._calculate_fan_in_and_fan_out(c.op.weight)[0]
                               for c in inh_cons]))

        # If there is recurrent excitation, use alpha scaling factor
        if nb_exc_rec >= 1:

            alpha = self.alpha

            delta_REC = np.sqrt((alpha * N_total_exc_rec) /
                                (N_total_exc_ff - alpha * N_total_exc_ff))
            delta_EI = (delta_REC * ebar_inh * N_total_inh * self.nu) / (delta_REC *
                                                                         ebar_exc * N_total_exc_ff * self.nu + ebar_exc * N_total_exc_rec * self.nu)

            lambda_exc_ff = self.xi * np.sqrt(2) * np.sqrt(self.nu) * np.sqrt(delta_EI**2 * ehat_exc * N_total_exc_rec + delta_REC**2 * (
                N_total_exc_ff * delta_EI**2 * ehat_exc + ehat_inh * N_total_inh)) / (theta * delta_EI * delta_REC)
            lambda_exc_rec = lambda_exc_ff * delta_REC
            lambda_inh = lambda_exc_ff * delta_EI

        # If not, scale automatically by number of synapses
        else:
            delta_EI = (N_total_inh * ebar_inh * self.nu) / \
                (N_total_exc * ebar_exc * self.nu)
            lambda_exc_ff = self.xi * np.sqrt(2) * np.sqrt(delta_EI**2 * N_total_exc *
                                                           self.nu * ehat_exc + N_total_inh * self.nu * ehat_inh) / (delta_EI * theta)
            lambda_exc_rec = lambda_exc_ff
            lambda_inh = lambda_exc_ff * delta_EI

        # Append parameters for all connections
        params = []
        for c in dst.afferents:
            if c.is_inhibitory:
                params.append(lambda_inh)
            elif c.is_recurrent:
                params.append(lambda_exc_rec)
            else:
                params.append(lambda_exc_ff)

        return params

    def initialize_layer(self, layer):

        # Loop through each population in this layer
        for neurons in layer.neurons:

            # Consider all afferents to this population
            # and compute weight parameters for each connection
            weight_params = self._get_weight_parameters_dst(neurons)

            # Initialize each connection
            for idx, connection in enumerate(neurons.afferents):
                # Read out parameters for weight distribution
                rate = weight_params[idx]
                # sample weights
                weights = self._get_weights(connection, rate)
                # set weights
                self._set_weights_and_bias(connection, weights)


class SpikeInitLogNormalInitializer(FluctuationDrivenNormalInitializer):

    def __init__(self,
                 sigma_u,
                 nu,
                 timestep,
                 epsilon_calc_mode='numerical',
                 alpha=0.9,
                 **kwargs):

        super().__init__(mu_u = 0.0,                # Fixed to balanced state
                         xi = 1 / sigma_u,
                         nu = nu,
                         timestep = timestep,
                         epsilon_calc_mode = epsilon_calc_mode,
                         alpha = alpha,
                         **kwargs)

    def _calc_epsilon(self, dst):
        """
        Calculates epsilon_bar and epsilon_hat, the integrals of the PSP kernel from a target
        neuron group `dst`
        """
        ebar_exc, ehat_exc = _get_epsilon(self.epsilon_calc_mode,
                                  dst.tau_mem,
                                  dst.tau_exc,
                                  self.timestep)

        ebar_inh, ehat_inh = _get_epsilon(self.epsilon_calc_mode,
                                  dst.tau_mem,
                                  dst.tau_inh,
                                  self.timestep)

        return ebar_exc, ehat_exc, ebar_inh, ehat_inh

    def _get_weights(self, connection, mu):
        shape = connection.op.weight.shape
        # sample weights
        weights = dists.LogNormal(mu, 1).sample(shape)
        return weights

    def initialize_connection(self, connection):
        """
        Initializes weights of a single `Connection` object
        """
        raise NotImplementedError

    def _get_weight_parameters_dst(self, dst):
        """
        Calculates weight parameters for all connections targeting a
        neuron group `dst`
        """

        theta = 1.0  # Theta (firing threshold) is hardcoded as in the LIFGroup code
        ebar_exc, ehat_exc, ebar_inh, ehat_inh = self._calc_epsilon(dst)

        # Read out some properties of the afferent connections
        inh_cons = [c for c in dst.afferents if c.is_inhibitory]
        exc_cons = [c for c in dst.afferents if not c.is_inhibitory]
        exc_ff_cons = [c for c in exc_cons if not c.is_recurrent]
        exc_rec_cons = [c for c in exc_cons if c.is_recurrent]

        nb_inh = len(inh_cons)
        nb_exc = len(exc_cons)
        nb_exc_ff = len(exc_ff_cons)
        nb_exc_rec = len(exc_rec_cons)

        # Assert that there is at least one excitatory and one inhibitory connection
        assert nb_inh >=1, 'each neuron group must have at least one inhibitory connection'
        assert nb_exc >=1, 'each neuron group must have at least one excitatory connection'

        # Sum of all inputs
        N_total_exc = int(sum([torch.nn.init._calculate_fan_in_and_fan_out(c.op.weight)[0]
                               for c in exc_cons]))
        N_total_exc_rec = int(sum([torch.nn.init._calculate_fan_in_and_fan_out(c.op.weight)[0]
                                   for c in exc_rec_cons]))
        N_total_exc_ff = int(sum([torch.nn.init._calculate_fan_in_and_fan_out(c.op.weight)[0]
                                  for c in exc_ff_cons]))
        N_total_inh = int(sum([torch.nn.init._calculate_fan_in_and_fan_out(c.op.weight)[0]
                               for c in inh_cons]))


        # If there is recurrent excitation, use alpha scaling factor
        if nb_exc_rec >= 1:

            alpha = self.alpha

            delta_REC = 1/2 * \
                np.log(N_total_exc_ff - alpha * N_total_exc_ff) - \
                np.log(alpha*N_total_exc_rec)
            delta_EI = 1/2 * np.log((ebar_exc * (np.exp(delta_REC) *
                                    N_total_exc_rec + N_total_exc_ff)) / (N_total_inh * ebar_inh))

            mu_exc_ff = 1/2 * (np.log(theta**2 / (self.xi**2 * (N_total_exc_rec*ehat_exc*self.nu*np.exp(
                2*delta_REC) + N_total_inh * ehat_inh * self.nu * np.exp(2*delta_EI) + N_total_exc_ff*self.nu*ehat_exc))) - 2)
            mu_exc_rec = mu_exc_ff + delta_REC
            mu_inh = mu_exc_ff + delta_EI

        # If not, scale automatically by number of synapses
        else:
            delta_EI = np.log((N_total_exc * self.nu * ebar_exc) /
                              (N_total_inh * self.nu * ebar_inh))
            mu_exc_ff = 1/2 * (np.log(theta**2 / (self.xi**2 * (N_total_exc*ehat_exc *
                               self.nu + N_total_inh * ehat_inh * self.nu * np.exp(2*delta_EI)))) - 2)
            mu_exc_rec = mu_exc_ff
            mu_inh = mu_exc_ff + delta_EI

        # Append parameters for all connections
        params = []
        for c in dst.afferents:
            if c.is_inhibitory:
                params.append(mu_inh)
            elif c.is_recurrent:
                params.append(mu_exc_rec)
            else:
                params.append(mu_exc_ff)

        return params


    def initialize_layer(self, layer):

        # Loop through each population in this layer
        for neurons in layer.neurons:

            # Consider all afferents to this population
            # and compute weight parameters for each connection
            weight_params = self._get_weight_parameters_dst(neurons)

            # Initialize each connection
            for idx, connection in enumerate(neurons.afferents):
                # Read out parameters for weight distribution
                mu = weight_params[idx]
                # sample weights
                weights = self._get_weights(connection, mu)
                # set weights
                self._set_weights_and_bias(connection, weights)
