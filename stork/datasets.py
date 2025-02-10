#!/usr/bin/env python3
import randman
import numpy as np
import scipy.sparse
import h5py
import pandas as pd

import string

import torch
import torch.utils.data


import threading

lock = threading.Lock()


def synchronized_open_file(*args, **kwargs):
    with lock:
        return h5py.File(*args, **kwargs)


def synchronized_close_file(self, *args, **kwargs):
    with lock:
        return self.close(*args, **kwargs)


def standardize(x, eps=1e-7):
    mi, _ = x.min(0)
    ma, _ = x.max(0)
    return (x - mi) / (ma - mi + eps)


def make_spaghetti_raster_dataset(
    nb_classes=10,
    nb_units=100,
    nb_steps=100,
    step_frac=1.0,
    nb_spikes=20,
    nb_samples=1000,
    alpha=2.0,
    shuffle=True,
    classification=True,
    seed=None,
):
    """Generates event based multi spike spaghetti manifold classification dataset.

    Args:
        nb_classes: The number of classes to generate
        nb_units: The number of units to assume
        nb_steps: The number of time steps to assume
        step_frac: Fraction of time steps from beginning of each to contain spikes (default 1.0)
        nb_spikes: The number of spikes to add to each input
        nb_samples: Number of samples from each manifold per class
        alpha: Randman smoothness parameter
        shuffe: Whether to shuffle the dataset
        classification: Whether to generate a classification (default) or regression dataset

    Returns:
        A tuple of data,labels. The data is structured as numpy array
        (sample x event x 2 ) where the last dimension contains
        the relative [0,1] (time,unit) coordinates and labels.
    """

    if seed is not None:
        np.random.seed(seed)

    rng_state = torch.random.get_rng_state()
    randman_seeds = np.random.randint(
        np.iinfo(np.int).max, size=(nb_classes, nb_spikes)
    )

    x = np.linspace(0, 1, nb_samples).reshape((-1, 1))
    data = []
    labels = []
    targets = []
    for k in range(nb_classes):
        submans = [
            randman.Randman(2, 1, alpha=alpha, seed=randman_seeds[k, i])
            for i in range(nb_spikes)
        ]
        spk = []
        for i, rm in enumerate(submans):
            y = rm.eval_manifold(x)
            y = standardize(y)
            spk.append(y.numpy())

        spk = np.array(spk)
        spk = spk.transpose((1, 0, 2))
        data.append(spk)
        labels.append(k * np.ones(len(spk)))
        targets.append(x)

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    targets = np.concatenate(targets, axis=0)

    if shuffle:
        idx = np.arange(len(data))
        np.random.shuffle(idx)
        data = data[idx]
        labels = labels[idx]
        targets = targets[idx]

    data[:, :, 0] *= nb_steps * step_frac
    data[:, :, 1] *= nb_units
    data = np.array(data, dtype=int)

    tmp = data
    data = []
    for d in tmp:
        data.append((d[:, 0], d[:, 1]))

    # restore torch.random state
    torch.random.set_rng_state(rng_state)

    data = torch.from_numpy(np.array(data))

    if classification:
        labels = torch.from_numpy(np.array(labels, dtype=int))
        return data, labels
    else:
        targets = torch.from_numpy(targets)
        return data, targets


def make_tempotron_dataset(
    nb_classes=2, nb_units=100, duration=1.0, step_frac=1.0, nb_samples=1000, seed=None
):
    """Generates event based generalized tempo randman classification dataset.

    In this dataset each unit fires a fixed number of spikes. So ratebased or spike count based decoding wont work.
    All the information is stored in the relative timing between spikes.
    For regression datasets the intrinsic manifold coordinates are returned for each target.

    Args:
        nb_classes: The number of classes to generate
        nb_units: The number of units to assume
        duration: Time in seconds of each stimulus
        nb_spikes: The number of spikes per unit
        nb_samples: Number of samples per class
        alpha: Randman smoothness parameter
        seed: The random seed (default: None)

    Returns:
        A tuple of data,labels. The data is structured as numpy array
        (sample x event x 2 ) where the last dimension contains
        the relative [0,1] (time,unit) coordinates and labels.
    """
    data = []
    labels = []

    if seed is not None:
        np.random.seed(seed)

    nb_data = nb_samples * nb_classes
    for k in range(nb_data):
        units = np.arange(nb_units)
        times = np.random.rand(nb_units)
        data.append((times, units))

    labels = np.arange(nb_data) % nb_classes

    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)

    return data, labels


def make_tempo_randman(
    nb_classes=10,
    nb_units=100,
    nb_steps=100,
    offset_frac=0.0,
    step_frac=1.0,
    dim_manifold=2,
    nb_spikes=2,
    nb_samples=1000,
    alpha=2.0,
    shuffle=True,
    classification=True,
    seed=None,
):
    """Generates event based generalized tempo randman classification dataset.

    In this dataset each unit fires a fixed number of spikes. So ratebased or spike count based decoding wont work.
    All the information is stored in the relative timing between spikes.
    For regression datasets the intrinsic manifold coordinates are returned for each target.

    Args:
        nb_classes: The number of classes to generate
        nb_units: The number of units to assume
        nb_steps: The number of time steps to assume
        offset_frac: Offset from beginning in fraction of time steps (default 0.0)
        step_frac: Fraction of time steps from beginning of each to contain spikes (default 1.0)
        nb_spikes: The number of spikes per unit
        nb_samples: Number of samples from each manifold per class
        alpha: Randman smoothness parameter
        shuffe: Whether to shuffle the dataset
        classification: Whether to generate a classification (default) or regression dataset
        seed: The random seed (default: None)

    Returns:
        A tuple of data,labels. The data is structured as numpy array
        (sample x event x 2 ) where the last dimension contains
        the relative [0,1] (time,unit) coordinates and labels.
    """

    data = []
    labels = []
    targets = []

    rng_state = torch.random.get_rng_state()

    if seed is not None:
        np.random.seed(seed)

    max_value = np.iinfo(np.int_).max
    randman_seeds = np.random.randint(max_value, size=(nb_classes, nb_spikes))

    for k in range(nb_classes):
        # dataset in the num. of dim. of the manifold
        x = np.random.rand(nb_samples, dim_manifold)
        # each spike in nb_spikes in each class is a Randman object
        # each Randman object corresponds to one manifold in the embedding space
        submans = [
            randman.Randman(
                embedding_dim=nb_units,
                manifold_dim=dim_manifold,
                alpha=alpha,
                seed=randman_seeds[k, i],
            )
            for i in range(nb_spikes)
        ]
        units = []
        times = []
        for i, rm in enumerate(submans):
            # convert the dataset to the num. of dim. of the embedding
            # each dim. of the embedding is one neuron, so each sample is a point on the manifold in embedding space
            # with one dim. corresponding to the firing times of one neuron
            y = rm.eval_manifold(x)
            y = standardize(y)
            units.append(
                np.repeat(np.arange(nb_units).reshape(1, -1), nb_samples, axis=0)
            )
            times.append(y.numpy())

        units = np.concatenate(units, axis=1)
        times = np.concatenate(times, axis=1)
        events = np.stack([times, units], axis=2)
        data.append(events)
        labels.append(k * np.ones(len(units)))
        targets.append(x)

    data = np.concatenate(data, axis=0)
    labels = np.array(np.concatenate(labels, axis=0), dtype=np.int_)
    targets = np.concatenate(targets, axis=0)

    if shuffle:
        idx = np.arange(len(data))
        np.random.shuffle(idx)
        data = data[idx]
        labels = labels[idx]
        targets = targets[idx]

    # convert normalized spike times into timesteps-based spike times
    data[:, :, 0] = nb_steps * (
        step_frac * (1.0 - offset_frac) * data[:, :, 0] + offset_frac
    )
    data = np.array(data, dtype=int)

    # restore torch.random state
    torch.random.set_rng_state(rng_state)

    data = [(torch.from_numpy(d[:, 0]), torch.from_numpy(d[:, 1])) for d in data]

    if classification:
        return data, labels
    else:
        return data, targets


def make_randman_halo(
    nb_classes=2,
    manifold_dim=1,
    embedding_dim=2,
    alpha=2,
    displacement=0.1,
    noise_ampl=1e-2,
    nb_samples=500,
):
    rm = randman.Randman(
        embedding_dim=embedding_dim, manifold_dim=manifold_dim, alpha=alpha
    )

    x = np.random.rand(nb_samples, manifold_dim)
    y = rm.eval_manifold(x)
    y = standardize(y)

    data = []
    labels = []
    for i in range(nb_classes):
        direction = torch.randn(nb_samples, embedding_dim)
        direction /= torch.norm(direction, p=2, dim=1, keepdim=True)
        noise = torch.randn(nb_samples, embedding_dim)
        points = y + i * direction * displacement + noise * noise_ampl
        data.append(points)
        labels.append(torch.ones(nb_samples) * i)

    data = torch.cat(data, 0)
    labels = torch.cat(labels, 0)
    return data, labels


def events2counts(data, nb_bins, mode="time"):
    """Converts the event based spike format to spike counts along time or space.

    Args:
        data: The dataset in event base time format (samples x spike events x 2 )
        nb_bins: The number of units or number of time steps depending on the mode.
               Should be an int.
        mode: Counts along time when mode="time" otherwise it will do a spatial
        count similar to population rate.

    Returns:
        An numpy array with spike counts (samples x units) or (samples x time steps)
        depending on mode.
    """

    event_x = data.copy()
    if mode == "space":
        event_x = np.array(event_x, dtype=int)[:, :, 0]
    else:
        if mode != "time":
            print(
                "Warning: Mode not recognized. Must be 'time' or 'space'. Assuming 'time'."
            )
        event_x = np.array(event_x, dtype=int)[:, :, 1]

    count_x = np.zeros((len(event_x), int(nb_bins)))
    for i, ev in enumerate(event_x):
        for j, un in enumerate(ev):
            count_x[i, int(un)] += 1

    return count_x


def dense2ras(densespikes, time_step=1e-3, concatenate_trials=True):
    """Returns ras spike format list of tuples (time, neuronid) or dense input data.

    Args:
    densespikes -- Either a matrix (time, neuron) of spikes or a rank 3 tensor (trial, time, neuron)
    time_step -- Time in seconds assumed per temporal biin

    Returns:
    A list of spikes in ras forma or when multiple trials are given as list of lists of spikes unless
    concatenate_trials is set to true in which case all trials will be concatenated.
    """

    if len(densespikes.shape) == 3:
        trials = []
        for spk in densespikes:
            trials.append(dense2ras(spk))
        if concatenate_trials:
            ras = []
            td = densespikes.shape[1]  # trial duration
            for k, trial in enumerate(trials):
                toff = np.zeros(trial.shape)
                toff[:, 0] = k * td * time_step
                ras.append(trial + toff)
            return np.concatenate(ras, axis=0)
        else:
            return trials
    elif len(densespikes.shape) == 2:
        ras = []
        aw = np.argwhere(densespikes > 0.0)
        for t, i in aw:
            ras.append((t * time_step, int(i)))
        return np.array(ras)
    else:
        print("Input array shape not understood.")
        raise ValueError


def ras2dense(ras, nb_steps, nb_units, time_step=1e-3):
    """Returns dense spike format matrix or tensor with 0s and 1s list from a ras input or a list of ras trials inputs.

    Args:
    ras: A list of spikes in ras format or a list of such lists for individual trials in ras format.
    time_step: Time in seconds assumed per temporal biin

    Returns: A dense spike array (time x unit) with zeros and ones or a tensor (trial x time x unit)
    """

    if type(ras) in (list, np.ndarray) and len(ras) > 0:
        ras = np.array(ras)
        if len(ras.shape) == 3:  # multiple trials
            A = np.zeros((len(ras), nb_steps, nb_units))
            for k, trial in enumerate(ras):
                for t, i in trial:
                    A[k, int(t / time_step), int(i)] = 1.0
            return A
        elif len(ras.shape) == 2:
            A = np.zeros((nb_steps, nb_units))
            for t, i in ras:
                A[int(t / time_step), int(i)] = 1.0
            return A
        else:
            print("Input data type not understood.")
            raise ValueError
    else:
        print("Cannot process empty list or array.")
        raise ValueError


def current2firing_time(x, tau=50e-3, thr=0.2, tmax=1.0, epsilon=1e-7):
    """Computes first firing time latency for a current input x assuming the charge time of a current based LIF neuron.

    Args:
    x -- The "current" values

    Keyword args:
    tau -- The membrane time constant of the LIF neuron to be charged
    thr -- The firing threshold value
    tmax -- The maximum time returned
    epsilon -- A generic (small) epsilon > 0

    Returns:
    Time to first spike for each "current" x
    """
    idx = x > thr
    T = torch.ones_like(x) * tmax
    T[idx] = tau * torch.log(x[idx] / (x[idx] - thr))
    return T


def firing_time2current(T, tau=50e-3, thr=0.2, tmax=1.0, epsilon=1e-7):
    """Inverts the computation done by firing_time as far as possible.

    Args:
    T -- The firing times

    Keyword Args:
    tau -- The membrane time constant of the LIF neuron to be charged
    thr -- The firing threshold value
    tmax -- The maximum time returned
    epsilon -- A generic (small) epsilon > 0

    Returns:
    Current value corresponding to the spike times.
    """
    e = np.exp(-T / tau)  # numerically more stable for T>0
    x = e * thr / (1.0 - e + epsilon)
    return x


def firing_time2dense(x, nb_steps=50, time_step=1e-3):
    """Converts firing times to a dense tensor of given dimensions.

    Args:
    x -- An object containing the firing times.

    Keyword args:
    nb_steps -- The number of time steps of the output dense tensor.
    time_step -- The timestep size in seconds.

    Returns:
    A dense matrix with zeros and ones corresponding to firing/spike times.
    """
    dat = np.zeros((x.shape[0], nb_steps, x.shape[1]))
    for k, sample in enumerate(x):
        for i, t in enumerate(sample):
            disct = int(t / time_step)
            if disct >= nb_steps:
                continue
            dat[k, disct, i] += 1
    return dat


def firing_time2sparse(x, nb_steps=50, time_step=1e-3):
    """Converts array of firing times to sparse matrix of spikes.

    Args:
    x -- An object containing the firing times.

    Keyword args:
    nb_steps -- The number of time steps of the output sparse matrix.
    time_step -- The timestep size in seconds.

    Returns:
    A sparse matrix with zeros and ones corresponding to firing/spike times.
    """

    nb_units = x.shape[1]
    dat = []
    for times in x:
        valid = np.array(times / time_step, dtype=int)
        idx = valid < nb_steps
        valid = valid[idx]
        row = np.array(valid, dtype=int)
        col = np.arange(nb_units, dtype=int)[idx]
        data = np.ones(nb_units)[idx]
        pat = scipy.sparse.coo_matrix((data, (row, col)), shape=(nb_steps, nb_units))
        dat.append(pat)
    return dat


def split_dataset(X, y, splits=[0.8, 0.2], shuffle=True):
    """Splits a dataset into training, validation and test set etc..

    Args:
        X: The data
        y: The labels
        splits: The split ratios (default=0.8,0.2)

    Returns:
        List containing tuples of (x_train, y_train), (x_test, y_test), etc ...
    """

    splits = np.array(splits)

    if (splits <= 0).any():
        raise AssertionError(
            "Split requires positive splitting ratios greater than zero."
        )
    splits /= splits.sum()

    if shuffle:
        idx = np.arange(len(X), dtype=np.int_)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

    start = 0
    sets = []
    for split in splits:
        idx_split = int(split * len(X))
        end = start + idx_split
        sets.append((X[start:end], y[start:end]))
        start = end
    return sets


# Set of functions to generate Poisson click trains as used in
# Brunton, B.W., Botvinick, M.M., and Brody, C.D. (2013). Rats and Humans Can
# Optimally Accumulate Evidence for Decision-Making. Science 340, 95–98.


def get_click_train(rate, duration=0.5):
    """Generates a Poisson click train of given rate and duration and returns
    the firing times."""
    intervals = np.random.exponential(
        scale=1.0 / rate, size=int(10 + 3 * rate * duration)
    )
    cumsum = np.cumsum(intervals)
    aw = np.argwhere(cumsum > duration).ravel()
    if len(aw):
        return cumsum[: aw[0]]
    else:
        print("Warning: Ran out of intervals during click train generation.")
        return cumsum


def get_click_trains(nb_samples=100, duration=0.5, sum_rate=40.0):
    """Generates a set of pairs of Poisson click trains for which the expected
    sum_rate is fixed (40Hz default)."""
    rL = sum_rate * np.random.rand(nb_samples)
    rR = sum_rate - rL
    data = []
    for s in range(nb_samples):
        a = get_click_train(rL[s], duration=duration)
        b = get_click_train(rR[s], duration=duration)
        data.append((a, b))
    return data


def get_poisson_click_dataset(nb_samples=100, duration=0.5, sum_rate=40.0):
    """Generates a dataset for supervised clasisifcation of click trains of
    given duration and sum_rate as used in Brunton, B.W., Botvinick, M.M., and
    Brody, C.D. (2013). Rats and Humans Can Optimally Accumulate Evidence for
    Decision-Making. Science 340, 95–98.

    Args:
        nb_samples: The number of samples
        duration: The duration of the spike trains in seconds
        sum_rate: The sum of the firing rates of the pairs of click trains.

    Returns:
        A doublet (data, labels) with data in RAS format list of (times,units).

    """

    trains = get_click_trains(
        nb_samples=nb_samples, duration=duration, sum_rate=sum_rate
    )
    data = []
    labels = []
    for a, b in trains:
        la = len(a)
        lb = len(b)
        l = 1 * (lb >= la)
        labels.append(l)
        units = np.zeros(la + lb)
        units[la:] = 1.0
        times = np.concatenate((a, b))
        idx = np.argsort(times)

        units = units[idx]
        times = times[idx]
        data.append((times, units))
    return data, labels


#######################################################################################################
# Class definitions
#######################################################################################################


class SpikingDataset(torch.utils.data.Dataset):
    """
    Provides a base class for all spiking dataset objects.
    """

    def __init__(
        self,
        nb_steps,
        nb_units,
        p_drop=0.0,
        p_insert=0.0,
        sigma_t=0.0,
        sigma_u=0.0,
        sigma_u_uniform=0.0,
        time_scale=1,
        data_augmentation=True, 
        # Before, data_augmentation was hardcoded to True in the __init__ method,
        # thus this defaults to True if not specified.
        # Note that with the default sigma_t=0.0, sigma_u=0.0, 
        # sigma_u_uniform=0.0, p_drop=0.0, p_insert=0.0,
        # the data_augmentation parameter has no effect.
    ):
        """
        This converter provides an interface for standard spiking datasets

        Args:
            p_drop: Probability of dropping a spike (default 0)
            p_insert: Probability of inserting a spurious spike in any time cell (default 0)
            sigma_t: Amplitude of time jitter added to each spike in bins (default 0)
            sigma_u: Amplitude of unit jitter added to each spike (default 0). The jitter is applied *after* unit scaling.
            sigma_u_uniform: Uniform noise amplitude added to all units (default 0). The jitter is applied *after* unit scaling.
            time_scale: Rescales the time-dimension (second dimension) of the dataset used to adjust to discrete time grid.
        """
        super().__init__()
        self.p_drop = p_drop
        self.p_insert = p_insert
        self.sigma_t = sigma_t
        self.sigma_u = sigma_u
        self.sigma_u_uniform = sigma_u_uniform
        self.time_scale = time_scale
        self.nb_steps = nb_steps
        self.nb_units = nb_units
        self.nb_insert = int(self.p_insert * self.nb_steps * self.nb_units)
        self.data_augmentation = data_augmentation

    def add_noise(self, times, units):
        """
        Expects lists of times and units as arguments and then adds spike noise to them.
        """

        if self.sigma_t:
            dt = torch.randn(len(times)) * self.sigma_t
            times = times + dt

        if self.sigma_u or self.sigma_u_uniform:
            if self.sigma_u:
                du = (torch.rand(len(units)) - 0.5) * self.sigma_u
                units = units + du
            if self.sigma_u_uniform:
                du = torch.randn(1) * self.sigma_u_uniform
                units = units + du

        if self.p_drop:
            rnd = torch.rand(len(times))
            idx = rnd > self.p_drop
            times = times[idx]
            units = units[idx]

        if self.p_insert:  # insert spurious extra spikes
            insert_times = (torch.rand(self.nb_insert) * self.nb_steps).long()
            insert_units = (torch.rand(self.nb_insert) * self.nb_units).long()
            times = torch.cat((times, insert_times))
            units = torch.cat((units, insert_units))

        return times, units

    def get_valid(self, times, units):
        """Return only the events that fall inside the input specs."""

        # Tag spikes which would otherwise fall outside of our self.nb_nb_steps
        idx = (times >= 0) & (times < self.nb_steps)

        idxu = (units >= 0) & (units < self.nb_units)
        idx = idx & idxu

        # Remove spikes which would fall outside of our nb_steps or nb_units
        times = times[idx]
        units = units[idx]

        return times, units

    def preprocess_events(self, times, units):
        """Apply data augmentation and filter out invalid events."""

        if self.data_augmentation:
            times, units = self.add_noise(times, units)

        times, units = self.get_valid(times, units)
        return times.long(), units.long()


class PoissonDataset(SpikingDataset):
    def __init__(
        self,
        dataset,
        nb_steps,
        nb_units,
        time_step,
        scale=1.0,
        p_drop=0.0,
        p_insert=0.0,
        sigma_t=0.0,
        start_frac=0.2,
        stop_frac=0.8,
        bg_act=0.0,
    ):
        """This dataset takes standard (vision) datasets as input and provides a time to first spike dataset.

        Args:
            dataset: The conventional analog dataset as a tuple (X,y)
            nb_steps: The number of time steps
            nb_units: The number of units should match the data
            time_step: The time step
            scale: Firing rate scale in Hz
            p_drop: Probability of dropping a spike (default 0)
            p_insert: Probability of inserting a spurious spike in any time cell (default 0)
            sigma_t: Amplitude of time jitter added to each spike in bins (default 0)
            start_frac: Relative fraction of duration at which stimulus turns on
            stop_frac: Relative fraction of duration when stimulus turns off
            bg_act: Background activity before and after stimulus in the same units as the dataset
        """

        self.scale = scale
        self.time_step = time_step
        X, y = dataset
        self.data = X
        self.labels = y.long()
        self.mul = scale * time_step
        self.bg_act = bg_act
        super().__init__(
            nb_steps, nb_units, p_drop=p_drop, p_insert=p_insert, sigma_t=sigma_t
        )
        self.start = int(start_frac * self.nb_steps)
        self.stop = int(stop_frac * self.nb_steps)

    def __len__(self):
        """Returns the total number of samples in dataset"""
        return len(self.data)

    def __getitem__(self, index):
        """Returns one sample of data"""
        p = self.mul * self.bg_act * torch.ones(self.nb_steps, self.nb_units)
        p[self.start : self.stop] = self.mul * self.data[index]
        X = (torch.rand(self.nb_steps, self.nb_units) < p).float()
        y = self.labels[index]
        return X, y


class RasDataset(SpikingDataset):
    def __init__(
        self,
        dataset,
        nb_steps,
        nb_units,
        p_drop=0.0,
        p_insert=0.0,
        sigma_t=0.0,
        time_scale=1,
        data_augmentation=False,
        dtype=torch.float32,
    ):
        """
        This converter provides an interface for standard Ras datasets to dense tensor format.

        Args:
            dataset: (data,labels) tuple where data is in RAS format
            p_drop: Probability of dropping a spike (default 0)
            p_insert: Probability of inserting a spurious spike in any time cell (default 0)
            sigma_t: Amplitude of time jitter added to each spike in bins (default 0)
            time_scale: Rescales the time-dimension (second dimension) of the dataset used to adjust to discrete time grid.
        """
        super().__init__(
            nb_steps,
            nb_units,
            p_drop=p_drop,
            p_insert=p_insert,
            sigma_t=sigma_t,
            time_scale=time_scale,
            data_augmentation=data_augmentation
        )

        data, labels = dataset

        if self.time_scale == 1:
            Xscaled = data
        else:
            Xscaled = []
            for times, units in data:
                times = self.time_scale * times
                idx = times < self.nb_steps
                Xscaled.append((times[idx], units[idx]))

        self.data = Xscaled
        self.labels = labels
        self.dtype = dtype
        if type(self.labels) == torch.tensor:
            self.labels = torch.cast(labels, dtype=dtype)

    def __len__(self):
        "Returns the total number of samples in dataset"
        return len(self.data)

    def __getitem__(self, index):
        "Returns one sample of data"

        times, units = self.data[index]
        times, units = self.preprocess_events(times, units)

        times = times.long()

        X = torch.zeros((self.nb_steps, self.nb_units), dtype=self.dtype)
        X[times, units] = 1.0
        y = self.labels[index]

        return X, y


class TextToRasDataset(RasDataset):
    def __init__(
        self,
        ds_name,
        path_to_csv,
        nb_steps,
        nb_units=None,
        chars_allowed=None,
        chars_convert=None,
        steps_bw_chars=0,
        fill_steps=False,
        p_drop=0.0,
        p_insert=0.0,
        sigma_t=0.0,
        time_scale=1,
    ):
        """
        This provides a way to initialize a RasDataset from a CSV dataset of text. It reads the CSV as a pandas
        DataFrame and converts the data into a RAS format.

        ds_name: name of the dataset path_to_csv: path to csv where the data is in the format: sentence, label
        chars_allowed: list of characters that will be recognized besides spaces, standard punctuations (as
        defined by string.punctuation), the 26 letters of the English alphabet and digits 0-9; all the other
        characters will be discarded during conversion
        chars_convert: dict for mapping characters, for example {'â': 'a'} will map 'â' to 'a' during conversion;
        note that all the values of the dict should be either in the default character list or the user-defined
        chars_allowed list
        steps_bw_chars(int): the number of steps to allow between two characters so that each letter
        does not come one after the other
        fill_steps(bool): whether the steps specified above should be filled with spike events
        """

        self.ds_name = ds_name
        ds = pd.read_csv(path_to_csv)
        col_name_samples, col_name_labels = ds.columns

        self.steps_bw_chars = steps_bw_chars
        self.fill_steps = fill_steps

        self.characters = (
            list(string.punctuation)
            + [" "]
            + list(string.digits)
            + list(string.ascii_lowercase)
        )
        # if the user defines any special characters, include in the recognizable character list
        if chars_allowed is not None:
            assert isinstance(chars_allowed, list)
            self.characters += chars_allowed

        self.chars_convert = chars_convert if chars_convert is not None else dict()
        # update character mapping to include default character list
        self.chars_convert.update(dict(zip(self.characters, self.characters)))
        # update character mapping to include those characters that have to be discarded
        unique_tokens_set = set(
            np.unique(np.array(list("".join(ds[col_name_samples]).lower())))
        )
        empty_mapping_keys = unique_tokens_set - set(chars_convert.keys())
        empty_mappings = dict(zip(empty_mapping_keys, [""] * len(empty_mapping_keys)))
        self.chars_convert.update(empty_mappings)

        # replace all letters with the correct mappings
        ds[col_name_samples] = ds[col_name_samples].apply(
            lambda x: pd.Series(list(x.lower())).map(self.chars_convert).to_list()
        )
        # recombine to get rid of the empty mappings
        ds[col_name_samples] = ds[col_name_samples].apply(lambda x: list("".join(x)))

        unit_mapping = dict(zip(self.characters, list(range(len(self.characters)))))

        if self.fill_steps:
            ds["units"] = ds[col_name_samples].apply(
                lambda x: pd.Series(x)
                .map(unit_mapping)
                .repeat(self.steps_bw_chars + 1)
                .to_list()
            )
            ds["times"] = ds.units.apply(lambda x: list(range(len(x))))
        else:
            ds["units"] = ds[col_name_samples].apply(
                lambda x: pd.Series(x).map(unit_mapping).to_list()
            )
            if steps_bw_chars != 0:
                ds["times"] = ds.units.apply(
                    lambda x: np.add(
                        np.arange(len(x)),
                        np.arange(0, len(x) * self.steps_bw_chars, self.steps_bw_chars),
                    )
                )
            else:
                ds["times"] = ds.units.apply(lambda x: np.arange(len(x)))

        # cast them all to torch.tensor
        ds.times = ds.times.apply(lambda x: torch.tensor(x))
        ds.units = ds.units.apply(lambda x: torch.tensor(x))

        input_data = list(zip(ds.times, ds.units))

        # convert labels from objects to int
        unique_labels = ds[col_name_labels].unique()
        label_mapping = dict(zip(unique_labels, np.arange(len(unique_labels))))
        labels = ds[col_name_labels].map(label_mapping).to_list()

        ds_to_parent = (input_data, labels)

        # number of input channels will be set to number of recognized characters
        nb_units_to_parent = nb_units if nb_units is not None else len(self.characters)

        # use produced dataset for initialising a RasDataset
        super().__init__(
            dataset=ds_to_parent,
            nb_steps=nb_steps,
            nb_units=nb_units_to_parent,
            p_drop=p_drop,
            p_insert=p_insert,
            sigma_t=sigma_t,
            time_scale=time_scale,
        )


class SpikeLatencyDataset(RasDataset):
    def __init__(
        self,
        data,
        nb_steps,
        nb_units,
        time_step=1e-3,
        tau=50e-3,
        thr=0.1,
        p_drop=0.0,
        p_insert=0.0,
        sigma_t=0.0,
    ):
        """This dataset takes standard (vision) datasets as input and provides a time to first spike dataset.

        Args:
            tau: Membrane time constant (default=50ms)
            thr: Firing threshold (default=0.1)
            p_drop: Probability of dropping a spike (default 0)
            p_insert: Probability of inserting a spurious spike in any time cell (default 0)
            sigma_t: Amplitude of time jitter added to each spike in bins (default 0)
        """

        self.time_step = time_step
        self.thr = thr
        self.tau = tau
        ras_data = self.prepare_data(
            data, tau_eff=tau / time_step, thr=thr, tmax=nb_steps
        )
        super().__init__(
            ras_data,
            nb_steps,
            nb_units,
            p_drop=p_drop,
            p_insert=p_insert,
            sigma_t=sigma_t,
            time_scale=1,
        )

    def prepare_data(self, data, tau_eff, thr, tmax):
        X, y = data
        nb_units = X.shape[1]

        # compute discrete firing times
        times = current2firing_time(X, tau=tau_eff, thr=self.thr, tmax=tmax).long()
        units = torch.arange(nb_units, dtype=torch.long)

        labels = y.long()
        ras = []
        for i in range(len(X)):
            idx = times[i] < tmax
            ras.append((times[i, idx], units[idx]))
        return (ras, labels)


class HDF5Dataset(SpikingDataset):
    def __init__(
        self,
        h5filepath,
        nb_steps,
        nb_units,
        p_drop=0.0,
        p_insert=0.0,
        sigma_t=0.0,
        sigma_u=0.0,
        sigma_u_uniform=0.0,
        time_scale=1.0,
        unit_scale=1.0,
        unit_permutation=None,
        preload=False,
        precompute_dense=False,
        sparse_output=False,
        coalesced=False,
    ):
        """
        This dataset acts as an interface for HDF5 datasets to dense tensor format.
        Per default this dataset class is not thread-safe unless used with the preload option.

        Args:
            h5filepath: The path and filename of the HDF5 file containing the data.
            p_drop: Probability of dropping a spike (default 0).
            p_insert: Probability of inserting a spurious spike in any time cell (default 0)
            sigma_t: Amplitude of time jitter added to each spike in bins (default 0). The jitter is applied *after* time scaling and in units of time bins.
            sigma_u: Amplitude of unit jitter added to each spike (default 0). The jitter is applied *after* unit scaling and in units of channels.
            sigma_u_uniform: Uniform noise amplitude added to channels
            time_scale: Rescales the time-dimension (second dimension) of the dataset used to adjust to discrete time grid.
            unit_scale: Rescales the time-dimension (second dimension) of the dataset used to adjust to discrete time grid.
            permute_units: Permute order of units before scaling
            preload: If set to true the datasets are first loaded into RAM instead of read from the HDF5 file directly.
            precompute_dense: If set to true the dense dataset is computed and stored in RAM (Warning! This may use a lot of RAM).
            sparse_output: If set to True, return sparse output tensor.
        """
        super().__init__(
            nb_steps,
            nb_units,
            p_drop=p_drop,
            p_insert=p_insert,
            sigma_t=sigma_t,
            sigma_u=sigma_u,
            sigma_u_uniform=sigma_u_uniform,
            time_scale=time_scale,
        )
        self.unit_scale = unit_scale
        self.sparse_output = sparse_output
        self.coalesced = coalesced
        self.precompute_dense = precompute_dense
        self.permutation = unit_permutation

        if preload:
            self.h5file = fileh = synchronized_open_file(h5filepath, mode="r")
            self.units = [x for x in fileh["spikes"]["units"]]
            self.times = [x for x in fileh["spikes"]["times"]]
            self.labels = [x for x in torch.tensor(fileh["labels"], dtype=torch.long)]
            synchronized_close_file(fileh)
        else:
            self.h5file = fileh = h5py.File(h5filepath, "r")
            self.units = fileh["spikes"]["units"]
            self.times = fileh["spikes"]["times"]
            self.labels = torch.tensor(fileh["labels"], dtype=torch.long)

        if precompute_dense:
            self.dataset = [self.get_dense(i) for i in range(len(self.labels))]

    def __len__(self):
        "Returns the total number of samples in dataset"
        return len(self.labels)

    def time_rescale(self, nb_steps, time_scale):
        self.nb_steps = nb_steps
        self.time_scale = time_scale

    def get_dense(self, index):
        "Convert a single sample from event-based format to dense."

        times = (torch.from_numpy(self.time_scale * self.times[index])).float()

        if self.permutation is None:
            units = np.array(self.unit_scale * self.units[index], dtype=int)
        else:
            units = np.array(
                self.unit_scale * self.permutation[self.units[index]], dtype=int
            )
        units = torch.from_numpy(units)

        times, units = self.preprocess_events(times, units)

        if self.sparse_output:
            y = self.labels[index]
            return (times, units), y
        else:
            if self.coalesced:
                # Slow but coalesced
                indices = torch.LongTensor(torch.stack([times, units], axis=1).T)
                values = torch.FloatTensor(torch.ones(len(times)))
                X = torch.sparse.FloatTensor(
                    indices, values, torch.Size([self.nb_steps, self.nb_units])
                ).to_dense()
            else:
                # Fast but not coalesced
                X = torch.zeros((self.nb_steps, self.nb_units), dtype=torch.float)
                X[times, units] = 1.0

            y = self.labels[index]
            return X, y

    def __getitem__(self, index):
        "Returns one sample of data"
        if self.precompute_dense:
            return self.dataset[index]
        else:
            return self.get_dense(index)


class DatasetView(torch.utils.data.Dataset):
    def __init__(self, dataset, elements):
        """
        This meta dataset provides a view onto an underlying Dataset by selecting a subset of elements specified in an index list.

        Args:
            dataset: The mother dataset instance
            elements: A list with indices of the data points in the mother dataset
        """
        super().__init__()
        self.dataset = dataset
        self.elements = elements

    def __len__(self):
        "Returns the total number of samples in dataset"
        return len(self.elements)

    def __getitem__(self, index):
        "Returns one sample of data"
        return self.dataset[self.elements[index]]


class TextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        text,
        nb_steps,
        nb_units,
        p_drop=0.0,
        p_insert=0.0,
        sigma_t=0.0,
        time_scale=1,
    ):
        """
        This converter provides an interface for text datasets to dense tensor character-prediction datasets.

        Args:
            data: The data in ras format
        """
        super().__init__()
        self.nb_steps = nb_steps
        self.nb_units = nb_units

        chars = tuple(set(text))
        int2char = dict(enumerate(chars))
        char2int = {ch: ii for ii, ch in int2char.items()}

        # Encode the text
        encoded = np.array([char2int[ch] for ch in text], dtype=np.int_)
        nb_samples = int(len(encoded) // nb_steps)
        encoded = encoded[: nb_samples * nb_steps]  # truncate
        self.data = encoded.reshape((nb_samples, nb_steps))
        self.times = np.arange(nb_steps)

    def __len__(self):
        "Returns the total number of samples in dataset"
        return len(self.data)

    def __getitem__(self, index):
        "Returns one sample of data"

        units = self.data[index]
        X = torch.zeros((self.nb_steps, self.nb_units))
        y = torch.zeros((self.nb_steps, self.nb_units))
        X[self.times, units] = 1.0
        y[:-1] = X[1:]

        return X, y
