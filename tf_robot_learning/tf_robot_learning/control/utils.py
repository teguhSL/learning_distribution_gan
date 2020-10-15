import numpy as np
import tensorflow as tf
from ..utils.basis_utils import build_fixed_psi

def get_canonical(nb_dim, nb_deriv=2, dt=0.01, return_op=True):
    A1d = np.zeros((nb_deriv, nb_deriv))

    for i in range(nb_deriv):
        A1d += np.diag(np.ones(nb_deriv - i), i) * np.power(dt, i) / np.math.factorial(i)

    B1d = np.zeros((nb_deriv, 1))
    for i in range(1, nb_deriv + 1):
        B1d[nb_deriv - i] = np.power(dt, i) / np.math.factorial(i)

    A, B = tf.constant(np.kron(A1d, np.eye(nb_dim)), dtype=tf.float32),\
		   tf.constant(np.kron(B1d, np.eye(nb_dim)), dtype=tf.float32)

    if return_op:
        return tf.linalg.LinearOperatorFullMatrix(A), tf.linalg.LinearOperatorFullMatrix(B)
    else:
        return A, B

def get_perturbation_seq(ndim=1, p_pert=0.01, pert_range=0.1, batch_size=10, horizon=100):
    p_push_prior = tf.concat([
        tf.log((1 - p_pert)) * tf.ones((batch_size, 1)),
        tf.log(p_pert) * tf.ones((batch_size, 1))
    ], axis=1)

    push_seq = tf.cast(tf.random.categorical(p_push_prior, horizon), tf.float32)[:, :, None] * \
               tf.random_normal(((batch_size, horizon, ndim)),
                                tf.zeros((batch_size, horizon, ndim)), pert_range)

    return push_seq

def get_push_seq(length=10, ndim=1, p_pert=0.01, pert_range=0.1, batch_size=10, horizon=100):
    k_basis = int(horizon / length * 2)

    _, h = build_fixed_psi(n_step=horizon, n_dim=ndim, n_state=k_basis,
                                          scale=.3 / k_basis)
    pert_seq = get_perturbation_seq(
        ndim=ndim, p_pert=p_pert, pert_range=pert_range, batch_size=batch_size, horizon=k_basis)

    push_seq = tf.reduce_sum(h[None, :, :, None] * pert_seq[:, None], axis=2)

    return push_seq
