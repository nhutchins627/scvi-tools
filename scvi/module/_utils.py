import torch
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras import backend as K
import sys


def compute_kernel(x, y, kernel='rbf', **kwargs):
    """
        Computes RBF kernel between x and y.
        # Parameters
            x: Tensor
                Tensor with shape [batch_size, z_dim]
            y: Tensor
                Tensor with shape [batch_size, z_dim]
        # Returns
            returns the computed RBF kernel between x and y
    """
    scales = kwargs.get("scales", [])
    if kernel == "rbf":
        x_size = K.shape(x)[0]
        y_size = K.shape(y)[0]
        dim = K.shape(x)[1]
        tiled_x = K.tile(K.reshape(x, K.stack([x_size, 1, dim])), K.stack([1, y_size, 1]))
        tiled_y = K.tile(K.reshape(y, K.stack([1, y_size, dim])), K.stack([x_size, 1, 1]))
        return K.exp(-K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, tf.float32))
    elif kernel == 'raphy':
        scales = K.variable(value=np.asarray(scales))
        squared_dist = K.expand_dims(squared_distance(x, y), 0)
        scales = K.expand_dims(K.expand_dims(scales, -1), -1)
        weights = K.eval(K.shape(scales)[0])
        weights = K.variable(value=np.asarray(weights))
        weights = K.expand_dims(K.expand_dims(weights, -1), -1)
        return K.sum(weights * K.exp(-squared_dist / (K.pow(scales, 2))), 0)
    elif kernel == "multi-scale-rbf":
        sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]

        beta = 1. / (2. * (K.expand_dims(sigmas, 1)))
        distances = squared_distance(x, y)
        s = K.dot(beta, K.reshape(distances, (1, -1)))

        return K.reshape(tf.reduce_sum(input_tensor=tf.exp(-s), axis=0), K.shape(distances)) / len(sigmas)


def squared_distance(x, y):  # returns the pairwise euclidean distance
    r = K.expand_dims(x, axis=1)
    return K.sum(K.square(r - y), axis=-1)


def compute_mmd(x, y, kernel, **kwargs):  # [batch_size, z_dim] [batch_size, z_dim]
    """
        Computes Maximum Mean Discrepancy(MMD) between x and y.
        # Parameters
            x: Tensor
                Tensor with shape [batch_size, z_dim]
            y: Tensor
                Tensor with shape [batch_size, z_dim]
        # Returns
            returns the computed MMD between x and y
    """
    x_kernel = compute_kernel(x, x, kernel=kernel, **kwargs)
    y_kernel = compute_kernel(y, y, kernel=kernel, **kwargs)
    xy_kernel = compute_kernel(x, y, kernel=kernel, **kwargs)
    return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)

def iterate(obj, func):
    """Iterates over an object and applies a function to each element."""
    t = type(obj)
    if t is list or t is tuple:
        return t([iterate(o, func) for o in obj])
    else:
        return func(obj) if obj is not None else None


def broadcast_labels(y, *o, n_broadcast=-1):
    """Utility for the semi-supervised setting.

    If y is defined(labelled batch) then one-hot encode the labels (no broadcasting needed)
    If y is undefined (unlabelled batch) then generate all possible labels (and broadcast other
    arguments if not None)
    """
    if not len(o):
        raise ValueError("Broadcast must have at least one reference argument")
    if y is None:
        ys = enumerate_discrete(o[0], n_broadcast)
        new_o = iterate(
            o,
            lambda x: x.repeat(n_broadcast, 1) if len(x.size()) == 2 else x.repeat(n_broadcast),
        )
    else:
        ys = torch.nn.functional.one_hot(y.squeeze(-1), n_broadcast)
        new_o = o
    return (ys,) + new_o


def enumerate_discrete(x, y_dim):
    """Enumerate discrete variables."""

    def batch(batch_size, label):
        labels = torch.ones(batch_size, 1, device=x.device, dtype=torch.long) * label
        return torch.nn.functional.one_hot(labels.squeeze(-1), y_dim)

    batch_size = x.size(0)
    return torch.cat([batch(batch_size, i) for i in range(y_dim)])


def masked_softmax(weights, mask, dim=-1, eps=1e-30):
    """Computes a softmax of ``weights`` along ``dim`` where ``mask is True``.

    Adds a small ``eps`` term in the numerator and denominator to avoid zero division.
    Taken from: https://discuss.pytorch.org/t/apply-mask-softmax/14212/15.
    Pytorch issue tracked at: https://github.com/pytorch/pytorch/issues/55056.
    """
    weight_exps = torch.exp(weights)
    masked_exps = weight_exps.masked_fill(mask == 0, eps)
    masked_sums = masked_exps.sum(dim, keepdim=True) + eps
    return masked_exps / masked_sums

def mmd(y, c, n_conditions, beta, boundary):
    """Initializes Maximum Mean Discrepancy(MMD) between every different condition.

       Parameters
       ----------
       n_conditions: integer
            Number of classes (conditions) the data contain.
       beta: float
            beta coefficient for MMD loss.
       boundary: integer
            If not 'None', mmd loss is only calculated on #new conditions.
       y: torch.Tensor
            Torch Tensor of computed latent data.
       c: torch.Tensor
            Torch Tensor of condition labels.

       Returns
       -------
       Returns MMD loss.
    """

    # partition separates y into num_cls subsets w.r.t. their labels c
    conditions_mmd = partition(y, c, n_conditions)
    loss = torch.tensor(0.0, device=y.device)
    if boundary is not None:
        for i in range(boundary):
            for j in range(boundary, n_conditions):
                if conditions_mmd[i].size(0) < 2 or conditions_mmd[j].size(0) < 2:
                    continue
                loss += mmd_loss_calc(conditions_mmd[i], conditions_mmd[j])
    else:
        for i in range(len(conditions_mmd)):
            if conditions_mmd[i].size(0) < 1:
                continue
            for j in range(i):
                if conditions_mmd[j].size(0) < 1 or i == j:
                    continue
                loss += mmd_loss_calc(conditions_mmd[i], conditions_mmd[j])

    return beta * loss
