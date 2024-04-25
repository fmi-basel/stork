import numpy as np
import time

import torch
import torch.nn as nn

import stork.nodes.base
from . import generators
from . import loss_stacks
from . import monitors


class RecurrentSpikingModel(nn.Module):
    def __init__(
        self,
        batch_size,
        nb_time_steps,
        nb_inputs,
        device=torch.device("cpu"),
        dtype=torch.float,
        sparse_input=False,
    ):
        super(RecurrentSpikingModel, self).__init__()
        self.batch_size = batch_size
        self.nb_time_steps = nb_time_steps
        self.nb_inputs = nb_inputs

        self.device = device
        self.dtype = dtype

        self.fit_runs = []

        self.groups = []
        self.connections = []
        self.devices = []
        self.monitors = []
        self.hist = []

        self.optimizer = None
        self.input_group = None
        self.output_group = None
        self.sparse_input = sparse_input

    def configure(
        self,
        input,
        output,
        loss_stack=None,
        optimizer=None,
        optimizer_kwargs=None,
        generator=None,
        time_step=1e-3,
        wandb=None,
        # added annealing options
        anneal_interval=0,
        anneal_step=0,
        anneal_start=0,
    ):
        self.input_group = input
        self.output_group = output
        self.time_step = time_step
        self.wandb = wandb
        self.anneal_interval = anneal_interval
        self.anneal_step = anneal_step
        self.anneal_start = anneal_start

        if loss_stack is not None:
            self.loss_stack = loss_stack
        else:
            self.loss_stack = loss_stacks.MaxOverTimeCrossEntropy()

        if generator is None:
            self.data_generator_ = generators.StandardGenerator()
        else:
            self.data_generator_ = generator

        # configure data generator
        self.data_generator_.configure(
            self.batch_size,
            self.nb_time_steps,
            self.nb_inputs,
            self.time_step,
            device=self.device,
            dtype=self.dtype,
        )

        for o in self.groups + self.connections:
            o.configure(
                self.batch_size,
                self.nb_time_steps,
                self.time_step,
                self.device,
                self.dtype,
            )

        if optimizer is None:
            optimizer = torch.optim.Adam

        if optimizer_kwargs is None:
            optimizer_kwargs = dict(lr=1e-3, betas=(0.9, 0.999))

        self.optimizer_class = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.configure_optimizer(self.optimizer_class, self.optimizer_kwargs)
        self.to(self.device)

    def time_rescale(self, time_step=1e-3, batch_size=None):
        """Saves the model then re-configures it with the old hyper parameters, but the new timestep.
        Then loads the model again."""
        if batch_size is not None:
            self.batch_size = batch_size
        saved_state = self.state_dict()
        saved_optimizer = self.optimizer_instance
        self.nb_time_steps = int(self.nb_time_steps * self.time_step / time_step)
        self.configure(
            self.input_group,
            self.output_group,
            loss_stack=self.loss_stack,
            generator=self.data_generator_,
            time_step=time_step,
            optimizer=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            wandb=self.wandb,
        )
        # Commenting this out will re-init the optimizer
        self.optimizer_instance = saved_optimizer
        self.load_state_dict(saved_state)

    def configure_optimizer(self, optimizer_class, optimizer_kwargs):
        if optimizer_kwargs is not None:
            self.optimizer_instance = optimizer_class(
                self.parameters(), **optimizer_kwargs
            )
        else:
            self.optimizer_instance = optimizer_class(self.parameters())

    def reconfigure(self):
        """Runs configure and replaces arguments with default from last run.

        This should reset the model an reinitialize all trainable variables.
        """
        if self.input_group is None or self.output_group is None:
            print(
                "Warning! No input or output group has been assigned yet. Run configure first."
            )
            return

        for o in self.groups + self.connections:
            o.configure(self.batch_size, self.device, self.dtype)

        # Re-init optimizer
        self.configure_optimizer(self.optimizer_class, self.optimizer_kwargs)

    def prepare_data(self, dataset):
        return self.data_generator_.prepare_data(dataset)

    def data_generator(self, dataset, shuffle=True):
        return self.data_generator_(dataset, shuffle=shuffle)

    def add_group(self, group):
        self.groups.append(group)
        self.add_module("group%i" % len(self.groups), group)
        return group

    def add_connection(self, con):
        self.connections.append(con)
        self.add_module("con%i" % len(self.connections), con)
        return con

    def add_monitor(self, monitor):
        self.monitors.append(monitor)
        return monitor

    def reset_states(self, batch_size=None):
        for g in self.groups:
            g.reset_state(batch_size)

    def evolve_all(self):
        for g in self.groups:
            g.evolve()
            g.clear_input()

    def apply_constraints(self):
        for c in self.connections:
            c.apply_constraints()

    def propagate_all(self):
        for c in self.connections:
            c.propagate()

    def execute_all(self):
        for d in self.devices:
            d.execute()

    def monitor_all(self):
        for m in self.monitors:
            m.execute()

    def compute_regularizer_losses(self):
        reg_loss = torch.zeros(1, device=self.device)
        for g in self.groups:
            reg_loss += g.get_regularizer_loss()
        for c in self.connections:
            reg_loss += c.get_regularizer_loss()
        return reg_loss

    def remove_regularizers(self):
        for g in self.groups:
            g.remove_regularizers()
        for c in self.connections:
            c.remove_regularizers()

    def run(self, x_batch, cur_batch_size=None, record=False):
        if cur_batch_size is None:
            cur_batch_size = len(x_batch)
        self.reset_states(cur_batch_size)
        self.input_group.feed_data(x_batch)
        for t in range(self.nb_time_steps):
            stork.nodes.base.CellGroup.clk = t
            self.evolve_all()
            self.propagate_all()
            self.execute_all()
            if record:
                self.monitor_all()
        self.out = self.output_group.get_out_sequence()
        return self.out

    def forward_pass(self, x_batch, cur_batch_size, record=False):
        # run recurrent dynamics
        us = self.run(x_batch, cur_batch_size, record=record)
        return us

    def get_example_batch(self, dataset, **kwargs):
        self.prepare_data(dataset)
        for batch in self.data_generator(dataset, **kwargs):
            return batch

    def get_total_loss(self, output, target_y, regularized=True):
        if type(target_y) in (list, tuple):
            target_y = [ty.to(self.device) for ty in target_y]
        else:
            target_y = target_y.to(self.device)

        self.out_loss = self.loss_stack(output, target_y)

        if regularized:
            self.reg_loss = self.compute_regularizer_losses()
            total_loss = self.out_loss + self.reg_loss
        else:
            total_loss = self.out_loss

        return total_loss

    def evaluate(self, test_dataset, train_mode=False, one_batch=False):
        self.train(train_mode)
        self.prepare_data(test_dataset)
        metrics = []
        for local_X, local_y in self.data_generator(test_dataset, shuffle=False):
            output = self.forward_pass(local_X, cur_batch_size=len(local_X))
            total_loss = self.get_total_loss(output, local_y)
            # store loss and other metrics
            metrics.append(
                [self.out_loss.item(), self.reg_loss.item()] + self.loss_stack.metrics
            )
            # evaluate only one batch (if needed for plotting)
            if one_batch:
                break

        return np.mean(np.array(metrics), axis=0)

    def regtrain_epoch(self, dataset, shuffle=True):
        self.train(True)
        self.prepare_data(dataset)
        metrics = []
        for local_X, local_y in self.data_generator(dataset, shuffle=shuffle):
            output = self.forward_pass(local_X, cur_batch_size=len(local_X))
            self.reg_loss = self.compute_regularizer_losses()

            total_loss = self.get_total_loss(output, local_y)
            loss = self.reg_loss

            # store loss and other metrics
            metrics.append(
                [self.out_loss.item(), self.reg_loss.item()] + self.loss_stack.metrics
            )

            # Use autograd to compute the backward pass.
            self.optimizer_instance.zero_grad()
            loss.backward()
            self.optimizer_instance.step()
            self.apply_constraints()

        return np.mean(np.array(metrics), axis=0)

    def train_epoch(self, dataset, shuffle=True):
        self.train(True)
        self.prepare_data(dataset)
        metrics = []
        for local_X, local_y in self.data_generator(dataset, shuffle=shuffle):
            output = self.forward_pass(local_X, cur_batch_size=len(local_X))
            total_loss = self.get_total_loss(output, local_y)

            # store loss and other metrics
            metrics.append(
                [self.out_loss.item(), self.reg_loss.item()] + self.loss_stack.metrics
            )

            # Use autograd to compute the backward pass.
            self.optimizer_instance.zero_grad()
            total_loss.backward()

            self.optimizer_instance.step()
            self.apply_constraints()

        return np.mean(np.array(metrics), axis=0)

    def get_metric_names(self, prefix="", postfix=""):
        metric_names = ["loss", "reg_loss"] + self.loss_stack.get_metric_names()
        return ["%s%s%s" % (prefix, k, postfix) for k in metric_names]

    def get_metrics_string(self, metrics_array, prefix="", postfix=""):
        s = ""
        names = self.get_metric_names(prefix, postfix)
        for val, name in zip(metrics_array, names):
            # nicer formatting for accuracy
            if "acc" in name:
                s = s + " {} = {:.1%},".format(name, val)
            else:
                s = s + " {} = {:.3e},".format(name, val)
        return s[:-1]

    def get_metrics_history_dict(self, metrics_array, prefix="", postfix=""):
        " Create metrics history dict. " ""
        s = ""
        names = self.get_metric_names(prefix, postfix)
        history = {name: metrics_array[:, k] for k, name in enumerate(names)}
        return history

    def prime(self, dataset, nb_epochs=10, verbose=True, wandb=None, offset=0):
        self.hist = []
        for ep in range(nb_epochs):
            t_start = time.time()
            ret = self.regtrain_epoch(dataset)
            self.hist.append(ret)

            if self.wandb is not None:
                self.wandb.log(
                    {key: value for (key, value) in zip(self.get_metric_names(), ret)},
                    step=ep + offset + 2,
                )

            if verbose:
                t_iter = time.time() - t_start
                print(
                    "%02i %s t_iter=%.2f" % (ep, self.get_metrics_string(ret), t_iter)
                )

        self.fit_runs.append(self.hist)
        history = self.get_metrics_history_dict(np.array(self.hist))
        return history

    # added validate option to fit (so that at some point, we don't need fit_validate anymore)
    def fit(
        self,
        dataset,
        valid_dataset=None,
        nb_epochs=10,
        verbose=True,
        logger=None,
        log_interval=10,
        monitor_spikes=False,
        anneal=False,
        offset=0,
    ):
        if valid_dataset is not None:
            validate = True
        self.hist_train = []
        if validate:
            self.hist_valid = []
        self.wall_clock_time = []

        if monitor_spikes:
            self.hist_ms = {}

        # For every epoch
        for ep in range(nb_epochs):
            t_start = time.time()

            # train
            self.train()
            ret_train = self.train_epoch(dataset)
            self.train(False)

            self.hist_train.append(ret_train)
            # validate
            if validate:
                ret_valid = self.evaluate(valid_dataset)
                self.hist_valid.append(ret_valid)

            # monitoring spike counts
            if monitor_spikes:
                self.train(False)
                in_group = self.input_group.get_flattened_out_sequence().detach().cpu()
                hid_groups = [
                    g.get_flattened_out_sequence().detach().cpu()
                    for g in self.groups[1:-1]
                ]

                if self.wandb is not None:
                    self.wandb.log(
                        {
                            "in_pop_spk_cnt": torch.mean(
                                torch.sum(in_group, dim=[1, 2])
                            ).item()
                        },
                        step=ep + offset + 2,
                    )
                    self.wandb.log(
                        {
                            "hid_pop_spk_cnt_"
                            + str(j): torch.mean(torch.sum(g, dim=[1, 2])).item()
                            for j, g in enumerate(hid_groups)
                        },
                        step=ep + offset + 2,
                    )

                if ep == 0:
                    self.hist_ms["in_spks"] = [
                        torch.mean(torch.sum(in_group, dim=1)).item()
                    ]
                    for gi, g in enumerate(hid_groups):
                        self.hist_ms["hid_{}_spks".format(gi)] = [
                            torch.mean(torch.sum(g, dim=1)).item()
                        ]
                else:
                    self.hist_ms["in_spks"].append(
                        torch.mean(torch.sum(in_group, dim=1)).item()
                    )
                    for gi, g in enumerate(hid_groups):
                        self.hist_ms["hid_{}_spks".format(gi)].append(
                            torch.mean(torch.sum(g, dim=1)).item()
                        )

            # measure time of iteration
            t_iter = time.time() - t_start

            # when using wandb -> log metrics
            if self.wandb is not None:
                if validate:
                    metrics = self.get_metric_names() + self.get_metric_names(
                        prefix="val_"
                    )
                    values = ret_train.tolist() + ret_valid.tolist()
                else:
                    metrics = self.get_metric_names()
                    values = ret_train.tolist()

                self.wandb.log(
                    {key: value for (key, value) in zip(metrics, values)},
                    step=ep + offset + 2,
                )

            # if there is a logger, log at every epoch
            if logger != None:
                if validate:
                    logger.info(
                        "%02i %s --%s t_iter=%.2f"
                        % (
                            ep,
                            self.get_metrics_string(ret_train),
                            self.get_metrics_string(ret_valid, prefix="val_"),
                            t_iter,
                        )
                    )
                else:
                    logger.info(
                        "%02i %s t_iter=%.2f"
                        % (ep, self.get_metrics_string(ret_train), t_iter)
                    )

            if verbose:
                if ep % log_interval == 0:
                    self.wall_clock_time.append(t_iter)
                    if validate:
                        print(
                            "%02i %s --%s t_iter=%.2f"
                            % (
                                ep,
                                self.get_metrics_string(ret_train),
                                self.get_metrics_string(ret_valid, prefix="val_"),
                                t_iter,
                            )
                        )
                    else:
                        print(
                            "%02i %s t_iter=%.2f"
                            % (ep, self.get_metrics_string(ret_train), t_iter)
                        )

            # when using annealing option
            if anneal:
                # TODO: check wheter the offset is correctly used
                if (ep + offset) >= self.anneal_start:
                    if (ep + offset - self.anneal_start) % self.anneal_interval == 0:
                        self.anneal_beta(ep)

        self.hist = np.array(self.hist_train)
        dict1 = self.get_metrics_history_dict(np.array(self.hist_train), prefix="")

        if not validate:
            history = {**dict1}
        else:
            self.hist = np.concatenate(
                (np.array(self.hist_train), np.array(self.hist_valid))
            )
            dict2 = self.get_metrics_history_dict(
                np.array(self.hist_valid), prefix="val_"
            )
            history = {**dict1, **dict2}

        # what is self.fit_runs?
        self.fit_runs.append(self.hist)

        # if monitor_spikes:
        #     return history, np.array(self.hist_ms)
        # else:
        return history

    def anneal_beta(self, ep, offset=0):
        """
        go through all spiking nonlinearities, change the betas and apply them again. This does not anneal escape noise parameters other than beta.
        """
        for g in range(1, len(self.groups) - 1):
            try:
                beta = self.groups[g].act_fn.surrogate_params["beta"]
                print("beta", beta)
                self.groups[g].act_fn.surrogate_params["beta"] = beta + self.anneal_step
                if "beta" in self.groups[g].act_fn.escape_noise_params.keys():
                    ebeta = self.groups[g].act_fn.escape_noise_params["beta"]
                    print("escape beta", ebeta)
                    self.groups[g].act_fn.escape_noise_params["beta"] = (
                        ebeta + self.anneal_step
                    )
                self.groups[g].spk_nl = self.groups[g].act_fn.apply
                print(
                    "annealed",
                    self.groups[g].act_fn.surrogate_params,
                    self.groups[g].act_fn.escape_noise_params,
                )
                self.wandb.log(
                    {"beta": beta, "noise beta": ebeta}, step=ep + offset + 2
                )

            except Exception as e:
                print(e)
                beta = self.groups[g].act_fn.beta
                print("beta", beta)
                self.groups[g].act_fn.beta = beta + self.anneal_step
                self.groups[g].spk_nl = self.groups[g].act_fn.apply
                print("annealed", self.groups[g].act_fn.beta)
                self.wandb.log({"beta": beta}, step=ep + offset + 2)

    # added annealing to fit_validate. TODO: add wandb support
    def fit_validate(
        self,
        dataset,
        valid_dataset,
        nb_epochs=10,
        verbose=True,
        log_interval=10,
        anneal=False,
        offset=0,
    ):
        self.hist_train = []
        self.hist_valid = []
        self.wall_clock_time = []

        # For every epoch
        for ep in range(nb_epochs):
            t_start = time.time()

            # train
            self.train()
            ret_train = self.train_epoch(dataset)
            self.train(False)

            # validate
            ret_valid = self.evaluate(valid_dataset)
            self.hist_train.append(ret_train)
            self.hist_valid.append(ret_valid)

            # when using wandb -> log metrics
            if self.wandb is not None:
                self.wandb.log(
                    {
                        key: value
                        for (key, value) in zip(
                            self.get_metric_names()
                            + self.get_metric_names(prefix="val_"),
                            ret_train.tolist() + ret_valid.tolist(),
                        )
                    },
                    step=ep + offset + 2,
                )

            # print metrics at given interval
            if verbose:
                if ep % log_interval == 0:
                    t_iter = time.time() - t_start
                    self.wall_clock_time.append(t_iter)
                    print(
                        "%02i %s --%s t_iter=%.2f"
                        % (
                            ep,
                            self.get_metrics_string(ret_train),
                            self.get_metrics_string(ret_valid, prefix="val_"),
                            t_iter,
                        )
                    )

            # when using annealing option
            if anneal:
                # TOO: check wheter the offset is correctly used
                if (ep + offset) >= self.anneal_start:
                    if (ep + offset - self.anneal_start) % self.anneal_interval == 0:
                        self.anneal_beta(ep)

        self.hist = np.concatenate(
            (np.array(self.hist_train), np.array(self.hist_valid))
        )
        self.fit_runs.append(self.hist)
        dict1 = self.get_metrics_history_dict(np.array(self.hist_train), prefix="")
        dict2 = self.get_metrics_history_dict(np.array(self.hist_valid), prefix="val_")
        history = {**dict1, **dict2}
        return history

    def get_probabilities(self, x_input):
        probs = []
        # we don't care about the labels, but want to use the generator
        fake_labels = torch.zeros(len(x_input))
        self.prepare_data((x_input, fake_labels))
        for local_X, local_y in self.data_generator((x_input, fake_labels)):
            output = self.forward_pass(local_X, cur_batch_size=len(local_y))
            tmp = torch.exp(self.loss_stack.log_py_given_x(output))
            probs.append(tmp)
        return torch.cat(probs, dim=0)

    def get_predictions(self, dataset):
        self.prepare_data(dataset)

        pred = []
        for local_X, _ in self.data_generator(dataset, shuffle=False):
            output = self.forward_pass(local_X, cur_batch_size=len(local_X))
            pred_labels = self.loss_stack.predict(output).cpu().numpy()
            pred.append(pred_labels)

        return np.concatenate(pred)

    def predict(self, data, train_mode=False):
        self.train(train_mode)
        if type(data) in [torch.Tensor, np.ndarray]:
            output = self.forward_pass(data, cur_batch_size=len(data))
            pred = self.loss_stack.predict(output)
            return pred
        else:
            self.prepare_data(data)
            pred = []
            for local_X, _ in self.data_generator(data, shuffle=False):
                data_local = local_X.to(self.device)
                output = self.forward_pass(data_local, cur_batch_size=len(local_X))
                pred.append(self.loss_stack.predict(output).detach().cpu())
            return torch.cat(pred, dim=0)

    def monitor(self, dataset):
        self.prepare_data(dataset)

        # Prepare a list for each monitor to hold the batches
        results = [[] for _ in self.monitors]
        for local_X, local_y in self.data_generator(dataset, shuffle=False):
            for m in self.monitors:
                m.reset()

            output = self.forward_pass(
                local_X, record=True, cur_batch_size=len(local_X)
            )

            for k, mon in enumerate(self.monitors):
                results[k].append(mon.get_data())

        return [torch.cat(res, dim=0) for res in results]

    def monitor_backward(self, dataset):
        """
        Allows monitoring of gradients with GradientMonitor
            - If there are no GradientMonitors, this runs the usual `monitor` method
            - Returns both normal monitor output and backward monitor output
        """

        # if there is a gradient monitor
        if any([isinstance(m, monitors.GradientMonitor) for m in self.monitors]):
            self.prepare_data(dataset)

            # Set monitors to record gradients
            gradient_monitors = [
                m for m in self.monitors if isinstance(m, monitors.GradientMonitor)
            ]
            for gm in gradient_monitors:
                gm.set_hook()

            # Prepare a list for each monitor to hold the batches
            results = [[] for _ in self.monitors]
            for local_X, local_y in self.data_generator(dataset, shuffle=False):
                for m in self.monitors:
                    m.reset()

                # forward pass
                output = self.forward_pass(
                    local_X, record=True, cur_batch_size=len(local_X)
                )

                # compute loss
                total_loss = self.get_total_loss(output, local_y)

                # Use autograd to compute the backward pass.
                self.optimizer_instance.zero_grad()
                total_loss.backward()

                # do not call an optimizer step as that would update the weights!

                # Retrieve data from monitors
                for k, mon in enumerate(self.monitors):
                    results[k].append(mon.get_data())

            # Turn gradient recording off
            for gm in gradient_monitors:
                gm.remove_hook()

            return [torch.cat(res, dim=0) for res in results]

        else:
            return self.monitor(dataset)

    def record_group_outputs(self, group, x_input):
        res = []
        # we don't care about the labels, but want to use the generator
        fake_labels = torch.zeros(len(x_input))
        self.prepare_data((x_input, fake_labels))
        for local_X, _ in self.data_generator((x_input, fake_labels)):
            output = self.forward_pass(local_X)
            res.append(group.get_out_sequence())
        return torch.cat(res, dim=0)

    def evaluate_ensemble(
        self, dataset, test_dataset, nb_repeats=5, nb_epochs=10, callbacks=None
    ):
        """Fits the model nb_repeats times to the data and returns evaluation results.

        Args:
            dataset: Training dataset
            test_dataset: Testing data
            nb_repeats: Number of repeats to retrain the model (default=5)
            nb_epochs: Train for x epochs (default=20)
            callbacks: A list with callbacks (functions which will be called as f(self) whose return value
                       is stored in a list and returned as third return value

        Returns:
            List of learning training histories curves
            and a list of test scores and if callbacks is not None an additional
            list with the callback results
        """
        results = []
        test_scores = []
        callback_returns = []
        for k in range(nb_repeats):
            print("Repeat %i/%i" % (k + 1, nb_repeats))
            self.reconfigure()
            self.fit(dataset, nb_epochs=nb_epochs, verbose=(k == 0))
            score = self.evaluate(test_dataset)
            results.append(np.array(self.hist))
            test_scores.append(score)
            if callbacks is not None:
                callback_returns.append([callback(self) for callback in callbacks])

        if callbacks is not None:
            return results, test_scores, callback_returns
        else:
            return results, test_scores

    def summary(self):
        """Print model summary"""

        print("\n# Model summary")
        print("\n## Groups")
        for group in self.groups:
            if group.name is None or group.name == "":
                print("no name, %s" % (group.shape,))
            else:
                print("%s, %s" % (group.name, group.shape))

        print("\n## Connections")
        for con in self.connections:
            print(con)


class DoubleInputRecSpikingModel(RecurrentSpikingModel):
    def __init__(
        self,
        batch_size,
        nb_time_steps,
        nb_inputs,
        nb_outputs,
        device=torch.device("cpu"),
        dtype=torch.float,
        sparse_input=False,
    ):
        super(RecurrentSpikingModel, self).__init__()
        self.batch_size = batch_size
        self.nb_time_steps = nb_time_steps
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs

        self.device = device
        self.dtype = dtype

        self.fit_runs = []

        self.groups = []
        self.connections = []
        self.devices = []
        self.monitors = []
        self.hist = []

        self.optimizer = None
        self.input_group = None
        self.output_group = None
        self.sparse_input = sparse_input

    def configure(
        self,
        input,
        output,
        loss_stack=None,
        optimizer=None,
        optimizer_kwargs=None,
        generator1=None,
        generator2=None,
        time_step=1e-3,
        wandb=None,
    ):
        self.input_group = input
        self.output_group = output
        self.time_step = time_step
        self.wandb = wandb

        if loss_stack is not None:
            self.loss_stack = loss_stack
        else:
            self.loss_stack = loss_stacks.TemporalCrossEntropyReadoutStack()

        if generator1 is None:
            self.data_generator1_ = generators.StandardGenerator()
        else:
            self.data_generator1_ = generator1

        if generator2 is None:
            self.data_generator2_ = generators.StandardGenerator()
        else:
            self.data_generator2_ = generator2

        # configure data generator
        self.data_generator1_.configure(
            self.batch_size,
            self.nb_time_steps,
            self.nb_inputs,
            self.time_step,
            device=self.device,
            dtype=self.dtype,
        )
        self.data_generator2_.configure(
            self.batch_size,
            self.nb_time_steps,
            self.nb_inputs,
            self.time_step,
            device=self.device,
            dtype=self.dtype,
        )

        for o in self.groups + self.connections:
            o.configure(
                self.batch_size,
                self.nb_time_steps,
                self.time_step,
                self.device,
                self.dtype,
            )

        if optimizer is None:
            optimizer = torch.optim.Adam

        if optimizer_kwargs is None:
            optimizer_kwargs = dict(lr=1e-3, betas=(0.9, 0.999))

        self.optimizer_class = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.configure_optimizer(self.optimizer_class, self.optimizer_kwargs)
        self.to(self.device)

    def monitor(self, datasets):
        # Prepare a list for each monitor to hold the batches
        results = [[] for _ in self.monitors]
        for (local_X1, local_y1), (local_X2, local_y2) in zip(
            self.data_generator1(datasets[0], shuffle=False),
            self.data_generator2(datasets[1], shuffle=False),
        ):
            for m in self.monitors:
                m.reset()

            local_X = torch.cat((local_X1, local_X2), dim=2)

            output = self.forward_pass(
                local_X,
                record=True,
                cur_batch_size=len(local_X),
            )

            for k, mon in enumerate(self.monitors):
                results[k].append(mon.get_data())

        return [torch.cat(res, dim=0) for res in results]

    def data_generator1(self, dataset, shuffle=True):
        return self.data_generator1_(dataset, shuffle=shuffle)

    def data_generator2(self, dataset, shuffle=True):
        return self.data_generator2_(dataset, shuffle=shuffle)

    def predict(self, datasets, train_mode=False):
        self.train(train_mode)
        print("predicting")
        if type(datasets) in [torch.Tensor, np.ndarray]:
            output = self.forward_pass(datasets, cur_batch_size=len(datasets))
            pred = self.loss_stack.predict(output)
            return pred
        else:
            # self.prepare_data(data)
            pred = []
            for (local_X1, local_y1), (local_X2, local_y2) in zip(
                self.data_generator1(datasets[0], shuffle=False),
                self.data_generator2(datasets[1], shuffle=False),
            ):
                local_X = torch.cat((local_X1, local_X2), dim=2)
                data_local = local_X.to(self.device)
                output = self.forward_pass(data_local, cur_batch_size=len(local_X))
                pred.append(self.loss_stack.predict(output).detach().cpu())
            return torch.cat(pred, dim=0)

    def train_epoch(self, dataset, shuffle=True):
        self.train(True)
        # self.prepare_data(dataset)
        metrics = []
        for (local_X1, local_y1), (local_X2, local_y2) in zip(
            self.data_generator1(dataset[0], shuffle=False),
            self.data_generator2(dataset[1], shuffle=False),
        ):
            local_X = torch.cat((local_X1, local_X2), dim=2)
            local_y1 = local_y1.unsqueeze(1)
            local_y2 = local_y2.unsqueeze(1)
            local_y = torch.cat((local_y1, local_y2), dim=1)

            output = self.forward_pass(local_X, cur_batch_size=len(local_X))
            # split output into parts corresponding to the first and second dataset
            output = torch.split(output, self.nb_outputs // 2, 2)
            total_loss = self.get_total_loss(output, local_y)

            # store loss and other metrics
            metrics.append(
                [self.out_loss.item(), self.reg_loss.item()] + self.loss_stack.metrics
            )

            # Use autograd to compute the backward pass.
            self.optimizer_instance.zero_grad()
            total_loss.backward()

            self.optimizer_instance.step()
            self.apply_constraints()

        return np.mean(np.array(metrics), axis=0)

    def evaluate(self, dataset, train_mode=False, one_batch=False):
        self.train(train_mode)
        # self.prepare_data(test_dataset)
        metrics = []
        for (local_X1, local_y1), (local_X2, local_y2) in zip(
            self.data_generator1(dataset[0], shuffle=False),
            self.data_generator2(dataset[1], shuffle=False),
        ):
            local_X = torch.cat((local_X1, local_X2), dim=2)
            local_y1 = local_y1.unsqueeze(1)
            local_y2 = local_y2.unsqueeze(1)
            local_y = torch.cat((local_y1, local_y2), dim=1)

            output = self.forward_pass(local_X, cur_batch_size=len(local_X))
            # split output into parts corresponding to the first and second dataset
            output = torch.split(output, self.nb_outputs // 2, 2)
            total_loss = self.get_total_loss(output, local_y)
            # store loss and other metrics
            metrics.append(
                [self.out_loss.item(), self.reg_loss.item()] + self.loss_stack.metrics
            )
            if one_batch:
                break

        return np.mean(np.array(metrics), axis=0)
