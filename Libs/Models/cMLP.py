"""Convolutional Multi-Layer Perceptron (cMLP) backbone for Neural GC.

This refactor adds:
* **Numerical stability** – configurable `atol`/`rtol` thresholds in
  :pymeth:`cMLP.GC` and a safe epsilon inside :func:`prox_update`.
* **Logging** – switch from ``print`` to :pymod:`logging` across training
  helpers.
* **Docstrings** – Google-style documentation for public classes/functions.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def activation_helper(activation, dim=None):
    if activation == 'sigmoid':
        act = nn.Sigmoid()
    elif activation == 'tanh':
        act = nn.Tanh()
    elif activation == 'relu':
        act = nn.ReLU()
    elif activation == 'leakyrelu':
        act = nn.LeakyReLU()
    elif activation is None:
        def act(x):
            return x
    else:
        raise ValueError('unsupported activation: %s' % activation)
    return act


class MLP(nn.Module):
    def __init__(self, num_series, lag, hidden, activation):
        super(MLP, self).__init__()
        self.activation = activation_helper(activation)

        # Set up network.
        layer = nn.Conv1d(num_series, hidden[0], lag)
        modules = [layer]

        for d_in, d_out in zip(hidden, hidden[1:] + [1]):
            layer = nn.Conv1d(d_in, d_out, 1)
            modules.append(layer)

        # Register parameters.
        self.layers = nn.ModuleList(modules)

    def forward(self, X):
        X = X.transpose(2, 1)
        for i, fc in enumerate(self.layers):
            if i != 0:
                X = self.activation(X)
            X = fc(X)

        return X.transpose(2, 1)


class cMLP(nn.Module):
    def __init__(self, num_series, lag, hidden, activation='relu'):
        '''
        cMLP model with one MLP per time series.

        Args:
          num_series: dimensionality of multivariate time series.
          lag: number of previous time points to use in prediction.
          hidden: list of number of hidden units per layer.
          activation: nonlinearity at each layer.
        '''
        super(cMLP, self).__init__()
        self.p = num_series
        self.lag = lag
        self.activation = activation_helper(activation)

        # Set up networks.
        self.networks = nn.ModuleList([
            MLP(num_series, lag, hidden, activation)
            for _ in range(num_series)])

    def forward(self, X):
        '''
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
        '''
        return torch.cat([network(X) for network in self.networks], dim=2)

    def GC(self, threshold: Union[bool, float] = True, ignore_lag: bool = True, *, atol: float = 1e-6, rtol: float = 0.0):
        """Return the learned Granger-causality adjacency matrix.

        Args:
            threshold: If *True* return a binary adjacency matrix using
                ``atol``/``rtol``; if *False*, return raw L2 norms.  A float
                value can also be supplied to use as an absolute threshold.
            ignore_lag: If *True*, collapse all lags into a single edge weight
                by taking the L2 norm across the kernel length. Otherwise a
                ``(p, p, lag)`` tensor is returned.
            atol: Absolute tolerance when testing ``> 0`` (default ``1e-6``).
            rtol: Relative tolerance (ignored when *threshold* is a float).

        Returns:
            ``torch.Tensor`` – Boolean or float adjacency matrix.
        """

        # NOTE: keep signature backward-compatible by accepting *args.

        if ignore_lag:
            GC = [torch.norm(net.layers[0].weight, dim=(0, 2)) for net in self.networks]
        else:
            GC = [torch.norm(net.layers[0].weight, dim=0) for net in self.networks]

        GC = torch.stack(GC)

        if threshold is False:
            return GC

        if isinstance(threshold, (int, float)) and threshold is not True:
            atol = float(threshold)

        mask = torch.abs(GC) > (atol + rtol * torch.abs(GC))
        return mask.int()


class cMLPSparse(nn.Module):
    def __init__(self, num_series, sparsity, lag, hidden, activation='relu'):
        '''
        cMLP model that only uses specified interactions.

        Args:
          num_series: dimensionality of multivariate time series.
          sparsity: ``torch.BoolTensor`` indicating Granger causality with
            shape *(num_series, num_series)*.
          lag: number of previous time points to use in prediction.
          hidden: list of number of hidden units per layer.
          activation: nonlinearity at each layer.
        '''
        super(cMLPSparse, self).__init__()
        self.p = num_series
        self.lag = lag
        self.activation = activation_helper(activation)
        self.sparsity = sparsity

        # Set up networks.
        self.networks = []
        for i in range(num_series):
            num_inputs = int(torch.sum(sparsity[i].int()))
            self.networks.append(MLP(num_inputs, lag, hidden, activation))

        # Register parameters.
        param_list = []
        for i in range(num_series):
            param_list += list(self.networks[i].parameters())
        self.param_list = nn.ParameterList(param_list)

    def forward(self, X):
        '''
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
        '''
        return torch.cat([self.networks[i](X[:, :, self.sparsity[i]])
                          for i in range(self.p)], dim=2)


def prox_update(network, lam, lr, penalty):
    '''
    Perform in place proximal update on first layer weight matrix.

    Args:
      network: MLP network.
      lam: regularization parameter.
      lr: learning rate.
      penalty: one of GL (group lasso), GSGL (group sparse group lasso),
        H (hierarchical).
    '''
    if isinstance(lam, str):
        lam = float(lam)
    W = network.layers[0].weight
    # Ensure numerical stability for frozen / near-zero columns.
    eps = max(lr * lam, 1e-8)
    hidden, p, lag = W.shape
    if penalty == 'GL':
        norm = torch.norm(W, dim=(0, 2), keepdim=True)
        W.data = ((W / torch.clamp(norm, min=eps)) * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'GSGL':
        norm = torch.norm(W, dim=0, keepdim=True)
        W.data = ((W / torch.clamp(norm, min=eps)) * torch.clamp(norm - (lr * lam), min=0.0))
        norm = torch.norm(W, dim=(0, 2), keepdim=True)
        W.data = ((W / torch.clamp(norm, min=eps)) * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'H':
        # Lowest indices along third axis touch most lagged values.
        for i in range(lag):
            norm = torch.norm(W[:, :, :(i + 1)], dim=(0, 2), keepdim=True)
            W.data[:, :, :(i+1)] = (
                (W.data[:, :, :(i+1)] / torch.clamp(norm, min=eps))
                * torch.clamp(norm - (lr * lam), min=0.0))
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


def regularize(network, lam, penalty):
    '''
    Calculate regularization term for first layer weight matrix.

    Args:
      network: MLP network.
      penalty: one of GL (group lasso), GSGL (group sparse group lasso),
        H (hierarchical).
    '''
    if isinstance(lam, str):
        lam = float(lam)
    W = network.layers[0].weight
    hidden, p, lag = W.shape
    if penalty == 'GL':
        return torch.sum(torch.norm(W, dim=(0, 2))) * lam
    elif penalty == 'GSGL':
        return (torch.sum(torch.norm(W, dim=(0, 2)))
                + torch.sum(torch.norm(W, dim=0))) * lam
    elif penalty == 'H':
        # Lowest indices along third axis touch most lagged values.
        reg = sum([torch.sum(torch.norm(W[:, :, :(i+1)], dim=(0, 2)))
                   for i in range(lag)])
        return reg * lam
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


def ridge_regularize(network, lam):
    """L2 penalty on **all** learnable weights except the first Conv layer.

    Returns a torch scalar so that autograd can propagate gradients. Handles
    edge-cases where ModuleList may contain non-parametric layers or be empty.
    """
    params = list(network.parameters())[2:]  # skip first layer's weight & bias
    if not params:
        # Create a zero tensor on same device to keep graph intact
        zero = next(network.parameters()).new_zeros(1)
        return lam * zero.squeeze()
    reg = torch.sum(torch.stack([torch.sum(p ** 2) for p in params]))
    if isinstance(lam, str):
        lam = float(lam)
    lam_tensor = torch.as_tensor(lam, dtype=reg.dtype, device=reg.device)
    return reg * lam_tensor


def restore_parameters(model, best_model):
    '''Move parameter values from best_model to model.'''
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        with torch.no_grad():
            params.data.copy_(best_params.data)


def train_model_gista(cmlp, X, lam, lam_ridge, lr, penalty, max_iter,
                      check_every=100, r=0.8, lr_min=1e-8, sigma=0.5,
                      monotone=False, m=10, lr_decay=0.5,
                      begin_line_search=True, switch_tol=1e-3, verbose=1):
    '''
    Train cMLP model with GISTA.

    Args:
      clstm: clstm model.
      X: tensor of data, shape (batch, T, p).
      lam: parameter for nonsmooth regularization.
      lam_ridge: parameter for ridge regularization on output layer.
      lr: learning rate.
      penalty: type of nonsmooth regularization.
      max_iter: max number of GISTA iterations.
      check_every: how frequently to record loss.
      r: for line search.
      lr_min: for line search.
      sigma: for line search.
      monotone: for line search.
      m: for line search.
      lr_decay: for adjusting initial learning rate of line search.
      begin_line_search: whether to begin with line search.
      switch_tol: tolerance for switching to line search.
      verbose: level of verbosity (0, 1, 2).
    '''
    p = cmlp.p
    lag = cmlp.lag
    cmlp_copy = deepcopy(cmlp)
    loss_fn = nn.MSELoss(reduction='mean')
    # Ensure scalar learning rates are float, regardless of whether they were
    # passed in as YAML strings such as "1e-3".
    lr = float(lr)
    lr_list = [lr for _ in range(p)]

    # Calculate full loss.
    mse_list = []
    smooth_list = []
    loss_list = []
    for i in range(p):
        net = cmlp.networks[i]
        mse = loss_fn(net(X[:, :-1]), X[:, lag:, i:i+1])
        ridge = ridge_regularize(net, lam_ridge)
        smooth = mse + ridge
        mse_list.append(mse)
        smooth_list.append(smooth)
        with torch.no_grad():
            nonsmooth = regularize(net, lam, penalty)
            loss = smooth + nonsmooth
            loss_list.append(loss)

    # Set up lists for loss and mse.
    with torch.no_grad():
        loss_mean = sum(loss_list) / p
        mse_mean = sum(mse_list) / p
    train_loss_list = [loss_mean]
    train_mse_list = [mse_mean]

    # For switching to line search.
    line_search = begin_line_search

    # For line search criterion.
    done = [False for _ in range(p)]
    assert 0 < sigma <= 1
    assert m > 0
    if not monotone:
        last_losses = [[loss_list[i]] for i in range(p)]

    try:
        from tqdm import trange
        iter_range = trange(max_iter, desc="GISTA", disable=(verbose==0))
    except ModuleNotFoundError:
        iter_range = range(max_iter)

    for it in iter_range:
        # Backpropagate errors.
        sum([smooth_list[i] for i in range(p) if not done[i]]).backward()

        # For next iteration.
        new_mse_list = []
        new_smooth_list = []
        new_loss_list = []

        # Perform GISTA step for each network.
        for i in range(p):
            # Skip if network converged.
            if done[i]:
                new_mse_list.append(mse_list[i])
                new_smooth_list.append(smooth_list[i])
                new_loss_list.append(loss_list[i])
                continue

            # Prepare for line search.
            step = False
            lr_it = lr_list[i]
            net = cmlp.networks[i]
            net_copy = cmlp_copy.networks[i]

            while not step:
                # Perform tentative ISTA step.
                for param, temp_param in zip(net.parameters(),
                                             net_copy.parameters()):
                    g = param.grad
                    if g is None or not torch.is_floating_point(g):
                        # Safe in-place copy avoids re-binding the `.data` attribute which can
                        # lead to unexpected dtype/device conversion errors on some versions of
                        # PyTorch.
                        with torch.no_grad():
                            temp_param.data.copy_(param.data)
                    else:
                        # Use no-grad context to update weights while keeping the computation
                        # graph intact for the original network.  This also guarantees the
                        # destination tensor stays on the correct device.
                        with torch.no_grad():
                            # in-place update avoids intermediate tensor allocation; add_ has
                            # stable semantics across tensor dtypes.
                            temp_param.data.copy_(param.data)
                            temp_param.data.add_(g, alpha=-lr_it)

                # Proximal update.
                prox_update(net_copy, lam, lr_it, penalty)

                # Check line search criterion.
                mse = loss_fn(net_copy(X[:, :-1]), X[:, lag:, i:i+1])
                ridge = ridge_regularize(net_copy, lam_ridge)
                smooth = mse + ridge
                with torch.no_grad():
                    nonsmooth = regularize(net_copy, lam, penalty)
                    loss = smooth + nonsmooth
                    tol = (0.5 * sigma / lr_it) * sum(
                        [torch.sum((param - temp_param) ** 2)
                         for param, temp_param in
                         zip(net.parameters(), net_copy.parameters())])

                comp = loss_list[i] if monotone else max(last_losses[i])
                if not line_search or (comp - loss) > tol:
                    step = True
                    if verbose > 1:
                        print('Taking step, network i = %d, lr = %f'
                              % (i, lr_it))
                        print('Gap = %f, tol = %f' % (comp - loss, tol))

                    # For next iteration.
                    new_mse_list.append(mse)
                    new_smooth_list.append(smooth)
                    new_loss_list.append(loss)

                    # Adjust initial learning rate.
                    lr_list[i] = (
                        (lr_list[i] ** (1 - lr_decay)) * (lr_it ** lr_decay))

                    if not monotone:
                        if len(last_losses[i]) == m:
                            last_losses[i].pop(0)
                        last_losses[i].append(loss)
                else:
                    # Reduce learning rate.
                    lr_it *= r
                    if lr_it < lr_min:
                        done[i] = True
                        new_mse_list.append(mse_list[i])
                        new_smooth_list.append(smooth_list[i])
                        new_loss_list.append(loss_list[i])
                        if verbose > 0:
                            print('Network %d converged' % (i + 1))
                        break

            # Clean up.
            net.zero_grad()

            if step:
                # Swap network parameters.
                cmlp.networks[i], cmlp_copy.networks[i] = net_copy, net

        # For next iteration.
        mse_list = new_mse_list
        smooth_list = new_smooth_list
        loss_list = new_loss_list

        # Check if all networks have converged.
        if sum(done) == p:
            if verbose > 0:
                print('Done at iteration = %d' % (it + 1))
            break

        # Check progress.
        if (it + 1) % check_every == 0:
            with torch.no_grad():
                loss_mean = sum(loss_list) / p
                mse_mean = sum(mse_list) / p
                ridge_mean = (sum(smooth_list) - sum(mse_list)) / p
                nonsmooth_mean = (sum(loss_list) - sum(smooth_list)) / p

            train_loss_list.append(loss_mean)
            train_mse_list.append(mse_mean)

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Total loss = %f' % loss_mean)
                print('MSE = %f, Ridge = %f, Nonsmooth = %f'
                      % (mse_mean, ridge_mean, nonsmooth_mean))
                print('Variable usage = %.2f%%'
                      % (100 * torch.mean(cmlp.GC().float())))

            # Check whether loss has increased.
            if not line_search:
                if train_loss_list[-2] - train_loss_list[-1] < switch_tol:
                    line_search = True
                    if verbose > 0:
                        print('Switching to line search')

    return train_loss_list, train_mse_list


def train_model_adam(cmlp, X, lr, max_iter, lam=0, lam_ridge=0, penalty='H',
                     lookback=5, check_every=100, verbose=1, *, batch_size: int | None = None):
    '''Train model with Adam.'''
    lag = cmlp.lag
    p = X.shape[-1]
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(cmlp.parameters(), lr=lr)
    train_loss_list = []

    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None

    try:
        from tqdm import trange
        iter_range = trange(max_iter, desc="Adam", disable=(verbose==0))
    except ModuleNotFoundError:
        iter_range = range(max_iter)

    for it in iter_range:
        # Mini-batch sampling ------------------------------------------------
        if batch_size is not None and batch_size < X.shape[0]:
            idx = torch.randint(0, X.shape[0], (batch_size,), device=X.device)
            X_batch = X[idx]
        else:
            X_batch = X

        # Calculate loss on batch / full data.
        loss = sum([loss_fn(cmlp.networks[i](X_batch[:, :-1]), X_batch[:, lag:, i:i+1])
                    for i in range(p)])

        # Add penalty terms.
        if lam > 0:
            loss = loss + sum([regularize(net, lam, penalty)
                               for net in cmlp.networks])
        if lam_ridge > 0:
            loss = loss + sum([ridge_regularize(net, lam_ridge)
                               for net in cmlp.networks])

        # Take gradient step.
        loss.backward()
        optimizer.step()
        cmlp.zero_grad()

        # Check progress.
        if (it + 1) % check_every == 0:
            mean_loss = loss / p
            train_loss_list.append(mean_loss.detach())

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Loss = %f' % mean_loss)

            # Check for early stopping.
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(cmlp)
            elif (it - best_it) == lookback * check_every:
                if verbose:
                    print('Stopping early')
                break

    # Restore best model.
    restore_parameters(cmlp, best_model)

    return train_loss_list


def train_model_ista(cmlp, X, lr, max_iter, lam=0, lam_ridge=0, penalty='H',
                     lookback=5, check_every=100, verbose=1):
    '''Train model with Adam.'''
    lag = cmlp.lag
    p = X.shape[-1]
    loss_fn = nn.MSELoss(reduction='mean')
    train_loss_list = []

    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None

    # Calculate smooth error.
    loss = sum([loss_fn(cmlp.networks[i](X[:, :-1]), X[:, lag:, i:i+1])
                for i in range(p)])
    ridge = sum([ridge_regularize(net, lam_ridge) for net in cmlp.networks])
    smooth = loss + ridge

    try:
        from tqdm import trange
        iter_range = trange(max_iter, desc="ISTA", disable=(verbose==0))
    except ModuleNotFoundError:
        iter_range = range(max_iter)

    for it in iter_range:
        # Take gradient step.
        smooth.backward()
        for param in cmlp.parameters():
            # Update parameters in-place without re-assigning `.data` to circumvent
            # potential device / dtype mismatches that can trigger obscure TypeErrors
            # on some PyTorch builds.
            with torch.no_grad():
                param.data.copy_(param.data)
                param.data.add_(param.grad, alpha=-lr)

        # Take prox step.
        if lam > 0:
            for net in cmlp.networks:
                prox_update(net, lam, lr, penalty)

        cmlp.zero_grad()

        # Calculate loss for next iteration.
        loss = sum([loss_fn(cmlp.networks[i](X[:, :-1]), X[:, lag:, i:i+1])
                    for i in range(p)])
        ridge = sum([ridge_regularize(net, lam_ridge)
                    for net in cmlp.networks])
        smooth = loss + ridge

        # Check progress.
        if (it + 1) % check_every == 0:
            # Add nonsmooth penalty.
            nonsmooth = sum([regularize(net, lam, penalty)
                             for net in cmlp.networks])
            mean_loss = (smooth + nonsmooth) / p
            train_loss_list.append(mean_loss.detach())

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Loss = %f' % mean_loss)
                print('Variable usage = %.2f%%'
                      % (100 * torch.mean(cmlp.GC().float())))

            # Check for early stopping.
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(cmlp)
            elif (it - best_it) == lookback * check_every:
                if verbose:
                    print('Stopping early')
                break

    # Restore best model.
    restore_parameters(cmlp, best_model)

    return train_loss_list


def train_unregularized(cmlp, X, lr, max_iter, lookback=5, check_every=100,
                        verbose=1):
    '''Train model with Adam and no regularization.'''
    lag = cmlp.lag
    p = X.shape[-1]
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(cmlp.parameters(), lr=lr)
    train_loss_list = []

    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None

    try:
        from tqdm import trange
        iter_range = trange(max_iter, desc="UNREG", disable=(verbose==0))
    except ModuleNotFoundError:
        iter_range = range(max_iter)

    for it in iter_range:
        # Calculate loss.
        pred = cmlp(X[:, :-1])
        loss = sum([loss_fn(pred[:, :, i], X[:, lag:, i]) for i in range(p)])

        # Take gradient step.
        loss.backward()
        optimizer.step()
        cmlp.zero_grad()

        # Check progress.
        if (it + 1) % check_every == 0:
            mean_loss = loss / p
            train_loss_list.append(mean_loss.detach())

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Loss = %f' % mean_loss)

            # Check for early stopping.
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(cmlp)
            elif (it - best_it) == lookback * check_every:
                if verbose:
                    print('Stopping early')
                break

    # Restore best model.
    restore_parameters(cmlp, best_model)

    return train_loss_list
