import numpy as np
import torch
import torch.utils.data


class DataGenerator:
    def __init__(self):
        """
        Args:
            shuffle: Whether to shuffle mini batches
        """

    def configure(self, batch_size, nb_steps, nb_units, time_step, device, dtype):
        """
        Args:
            batch_size: The batch size
            nb_steps: The integer number of timesteps
            nb_units: Number of units
            device: The torch device to use
            dtype: The torch dtype to use
            time_step: The simulation time step in seconds
        """
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.nb_units = nb_units
        self.device = device
        self.dtype = dtype
        self.time_step = time_step

    def prepare_data(self, dataset):
        pass

    def __call__(self, dataset, shuffle=True):
        raise NotImplementedError


class StandardGenerator(DataGenerator):
    """
    Provides a new transitional generator class based on the Torch Dataset/Loader formalism.
    """

    def __init__(self, nb_workers=1, persistent_workers=True, worker_init_fn=None):
        self.nb_workers = nb_workers
        self.persistent_workers = persistent_workers
        self.worker_init_fn = worker_init_fn

    def __call__(self, dataset, shuffle=True):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.nb_workers,
            persistent_workers=self.persistent_workers,
            worker_init_fn=self.worker_init_fn,
        )


class SparseGenerator(DataGenerator):
    """Provides a new transitional generator class based on the Torch Dataset/Loader formalism.

    shuffle: Whether to shuffle minibaches
    """

    def __init__(self, nb_workers=1):
        self.nb_workers = nb_workers

    def get_sparse_collate(self):
        """Returns collate function"""

        def collate_function(data_labels):
            """Collates list of tuples with (indices, values) to sparse tensor."""
            data = [x for x, _ in data_labels]
            labels = [l for _, l in data_labels]
            batch_indices = np.concatenate(
                [k * np.ones(len(x[0]), dtype=int) for k, x in enumerate(data)]
            )
            time_indices = np.concatenate([ind for ind, _ in data])
            neuron_indices = torch.FloatTensor(np.concatenate([ind for _, ind in data]))
            indices = torch.LongTensor(
                np.stack([batch_indices, neuron_indices], axis=0)
            )
            values = torch.FloatTensor(
                np.concatenate([np.ones(len(x[0])) for x in data])
            )
            return [
                (indices[:, time_indices == t], values[time_indices == t])
                for t in range(self.nb_steps)
            ], torch.stack(labels, dim=0)

        return collate_function

    def __call__(self, dataset, shuffle=True):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.nb_workers,
            collate_fn=self.get_sparse_collate(),
        )
