import numpy as np
import matplotlib.pyplot as plt
import stork.utils
from tqdm import trange

def get_poisson_spikes(N, nu, duration, dt):
    """Generates a set of Poisson-distributed spikes for $N$ neurons that fire at frequency $\nu$.

    Args:
        N (int): Number of input neurons
        nu (float): Firing rate of input neurons in Hz
        duration (float): Duration of simulation in seconds
        dt (float): Simulation time step in seconds

    Returns:
        np.array: Array containing Poisson distributed spikes
    """
    timesteps = int(duration / dt)
    spikes = np.zeros((timesteps, N))
    spikeprob_per_timestep = nu * dt

    for t in range(timesteps):
        spikes[t] = np.random.rand(N) < spikeprob_per_timestep

    return spikes


def plot_spikeraster(spikes, timestep, axis, max_time=1, max_neurons=500):
    """Generates a spike raster plot

    Args:
        spikes: Binary array containing spikes
        timestep (float): Simulation time step.
        axis: Axis to plot on.
        max_time (int, optional): Maximal number of time steps to display. Defaults to 1.
        max_neurons (int, optional): Maximal number of neurons to display. Defaults to 500.

    Returns:
        _type_: _description_
    """
    max_timesteps = int(max_time / timestep)  # max timestep calculation

    spiketimes = []  # convert to spike times
    for t in range(max_timesteps):
        spiking_neurons = np.where(spikes[t] == 1)[0]
        for idx in spiking_neurons:
            if idx <= max_neurons:
                spiketimes.append((t, idx))
    spiketimes = np.array(spiketimes)

    time_axis = (
        spiketimes[:, 0] * timestep
    )  # get x and y axis: multiply by timestep to get seconds
    neuron_axis = spiketimes[:, 1]

    axis.scatter(time_axis, neuron_axis, s=1.5, c="k")  # plot
    axis.set_ylabel("neuron idx")
    axis.set_xlabel("time [s]")

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    return axis


def epsilon_analytical(tau_mem, tau_syn):
    epsilon_bar = tau_syn
    epsilon_hat = (tau_syn**2) / (2 * (tau_syn + tau_mem))

    return epsilon_bar, epsilon_hat


def epsilon_numerical(tau_mem, tau_syn, timestep):
    kernel = stork.utils.get_lif_kernel(tau_mem, tau_syn, timestep)
    epsilon_bar = kernel.sum() * timestep
    epsilon_hat = (kernel**2).sum() * timestep

    return epsilon_bar, epsilon_hat


def get_epsilon(tau_mem, tau_syn, calc_mode="numerical", timestep=1e-3):
    if calc_mode == "analytical":
        return epsilon_analytical(tau_mem, tau_syn)
    elif calc_mode == "numerical":
        return epsilon_numerical(tau_mem, tau_syn, timestep)
    else:
        raise ValueError("invalid calc mode for epsilon")


def get_sigma_w(theta, xi, N, nu, mu_U, epsilon_hat, mu_w):
    """Calculate standard deviation of the weight distribution, that can be used for initialization.

    Args:
        theta (float): Threshold.
        xi (float): Initialization parameter, usually 1<xi<3.
        N (int): Number of inputs.
        nu (float): Input firing rate.
        mu_U (float): Target mean membrane potential.
        epsilon_hat (float): Integral of the squared PSP-kernel.
        mu_w (float): Mean of the weight distribution.

    Returns:
        float: Standard deviation of the weight distribution.
    """

    sigma2_w = 1 / (N * nu * epsilon_hat) * ((theta - mu_U) / xi) ** 2 - mu_w**2
    sigma_w = np.sqrt(sigma2_w)
    return sigma_w


def get_mu_w(N, nu, mu_U, epsilon_bar):
    """Calculate the mean of the weight distribution, that can be used for initialization.

    Args:
        N (int): Number of input neurons.
        nu (float): Input firing rate.
        mu_U (float): Target mean membrane potential.
        epsilon_bar (float): Integral of the PSP-Kernel.

    Returns:
        float: Mean of the weight distribution.
    """
    mu_w = mu_U / (N * nu * epsilon_bar)
    return mu_w


def sample_normal(N, mu, sigma):
    return mu + np.random.randn(N) * sigma


def get_w_params(N, nu, theta, mu_U, xi, tau_mem, tau_syn, eps_calc_mode="numerical"):
    """
    Get mean and standard deviation for the synaptic weight distribution based on
    a target in the fluctuation-driven reigme (defined by mu_U and xi)

    Args:
        N (int): Number of inputs.
        nu (float, optional): Input firing rate. Defaults to None.
        theta (float, optional): Threshold. Defaults to None.
        mu_U (float, optional): Target mean membrane potential. Defaults to None.
        xi (float, optional): Initialization parameter, usually 1<xi<3. Defaults to None.
        tau_mem (float, optional): Membrane time constant. Defaults to None.
        tau_syn (float, optional): Synaptic time constant. Defaults to None.
        eps_calc_mode (str, optional): Calculation mode of epsilons. Defaults to "numerical".

    Returns:
        weights
    """
    epsilon_bar, epsilon_hat = get_epsilon(
        calc_mode=eps_calc_mode, tau_mem=tau_mem, tau_syn=tau_syn
    )
    mu_w = get_mu_w(N, nu, mu_U, epsilon_bar)
    sigma_w = get_sigma_w(theta, xi, N, nu, mu_U, epsilon_hat, mu_w)

    return mu_w, sigma_w


def plot_histogram_with_gaussian(
    data,
    mu,
    sigma,
    axis,
    color,
    linecolor,
    bins=40,
    orientation="vertical",
    minmax=None,
    valrange=None,
    linewidth=1.0,
    edgecolor="white",
    alpha=1,
):

    if minmax is None:
        minmax = (np.round(-4 * sigma, 1), np.around(4 * sigma, 1))

    xax = np.arange(minmax[0], minmax[1], 0.01)
    axis.hist(
        data,
        bins=bins,
        color=color,
        density=True,
        orientation=orientation,
        range=valrange,
        label="Simulation",
        edgecolor=edgecolor,
        alpha=alpha
    )

    if orientation == "horizontal":
        pdf = gaussian_pdf(xax, mu, sigma)
        axis.plot(pdf, xax, color=linecolor, lw=linewidth, label="Theory")

        # remove bottom axis
        axis.get_xaxis().set_visible(False)
        axis.spines["bottom"].set_visible(False)
        axis.set_ylim(minmax[0], minmax[1])
        axis.set_yticks([minmax[0], mu, minmax[1]])

    else:
        pdf = gaussian_pdf(xax, mu, sigma)
        axis.plot(xax, pdf, color=linecolor, lw=linewidth, label="Theory")

        # remove left axis
        axis.get_yaxis().set_visible(False)
        axis.spines["left"].set_visible(False)

        axis.set_xlim(minmax[0], minmax[1])
        axis.set_xticks([minmax[0], mu, minmax[1]])

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    return axis


def gaussian_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def get_membrane(I, tau_mem, theta, dt, refractory_period=0, spike_amp=2):
    """Compute the membrane potential.

    Args:
        I: Input current.
        tau_mem (float): Membrane time constant.
        theta (float): Threshold.
        dt (float): Simulation time step.
        refractory_period (float, optional): Refractory period. Defaults to 0.
        spike_amp (float, optional): Height of a spike. Defaults to 2.

    Returns:
        Membrane potential.
    """
    u = np.zeros(I.shape)
    decay = np.exp(-dt / tau_mem)
    refrac = 0


    if len(u.shape) == 1:
        for t in range(1, u.shape[0]):
            if refrac > 0:  # Implement refractory period
                u[t] = 0
            else:
                u[t] = u[t - 1] * decay + (1 - decay) * I[t]  # update
                if u[t] >= theta:  # spikes
                    u[t] = spike_amp
                    refrac = refractory_period
            refrac -= dt  # decay refrac
    else:
        for n in trange(u.shape[1]):
            refrac = 0
            for t in range(1, u.shape[0]):
                if refrac > 0:  # Implement refractory period
                    u[t, n] = 0
                else:
                    u[t, n] = u[t - 1, n] * decay + (1 - decay) * I[t, n]  # update
                    if u[t, n] >= theta:  # spikes
                        u[t, n] = spike_amp
                        refrac = refractory_period
                refrac -= dt  # decay refrac
    return u


def get_current(spikes, weights, tau_syn, dt):
    """Compute synaptic currents.

    Args:
        spikes: Incoming spike trains.
        weights: Synaptic weights.
        tau_syn (float): Synaptic time constant.
        dt (float): Simulation time step.

    Returns:
        Synaptic current.
    """
    currents = np.zeros(spikes.shape)
    decay = np.exp(-dt / tau_syn)

    for t in range(1, currents.shape[0]):
        currents[t] = currents[t - 1] * decay + spikes[t]

    currents = currents @ weights.T
    return currents


def get_mu_U(N, nu, epsilon_bar, mu_w):
    return N * nu * epsilon_bar * mu_w


def get_sigma_U(N, nu, epsilon_hat, mu_w, sigma_w):
    return np.sqrt(N * (mu_w**2 + sigma_w**2) * nu * epsilon_hat)


def get_mempot_range(membrane):

    absmax = np.abs(membrane).max().round(1)
    return (-absmax, absmax)


def get_membrane_without_spikes(I, tau_mem, dt):
    """
    Only implement leaky membrane, without refractory period or spike reset
    """
    u = np.zeros(I.shape)
    decay = np.exp(-dt / tau_mem)

    for t in range(1, u.shape[0]):
        u[t] = u[t - 1] * decay + (1 - decay) * I[t]

    return u


def run_simulation(
    mu_U,
    xi,
    tau_mem,
    tau_syn,
    N,
    nu,
    duration,
    dt,
    theta,
    spike_amp,
    calc_mode="numerical",
    refractory_period=2e-3,
):
    # Get initial weights
    mu_w, sigma_w = get_w_params(
        N, nu, theta, mu_U, xi, tau_mem, tau_syn, eps_calc_mode=calc_mode
    )
    weights = sample_normal(N, mu_w, sigma_w)

    # Get spikes, currents and membrane potential
    spikes = get_poisson_spikes(N=N, nu=nu, duration=duration, dt=dt)
    currents = get_current(spikes=spikes, weights=weights, tau_syn=tau_syn, dt=dt)
    membrane = get_membrane_without_spikes(I=currents, tau_mem=tau_mem, dt=dt)

    # Get membrane potential considering reset and output firing rate
    out_with_spikes = get_membrane(
        I=currents,
        tau_mem=tau_mem,
        theta=theta,
        dt=dt,
        refractory_period=refractory_period,
        spike_amp=spike_amp,
    )
    out_without_spikes = out_with_spikes.copy()
    out_without_spikes[out_without_spikes == spike_amp] = 0

    # Analytical calculation of mu_U and sigma_u
    epsilon_bar, epsilon_hat = get_epsilon(
        calc_mode="analytical", tau_mem=tau_mem, tau_syn=tau_syn
    )

    mu_U = get_mu_U(N=N, nu=nu, epsilon_bar=epsilon_bar, mu_w=mu_w)
    sigma_U = get_sigma_U(
        N=N, nu=nu, epsilon_hat=epsilon_hat, mu_w=mu_w, sigma_w=sigma_w
    )

    return membrane, out_without_spikes, mu_U, sigma_U
