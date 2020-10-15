import tensorflow as tf

import numpy as np
from ...distributions import *
from ...utils.param_utils import make_cov, make_loc


def make_rollout_samples_is(xi0, f, pi, u_dim, T=100, batch_shape=None, t0=0,
							time_first=False):
	"""
	Execute rollouts of trajectories with importance sampling
	Can be used when f or pi is a mixture of experts. Instead of sampling the mixture
	components as given as gate(x), we sample according to exp(g(log(gate(x)))) where g change
	the probabilty and weights are applied on the trajectories.

	:param xi0: 	Initial state [batch_shape, xi_dim] or [xi_dim]
	:param f: 		Dynamic model (xi [bs, xi_dim], u [bs, u_dim]) -> (xi [bs, xi_dim]), (ws [bs,])
	:param pi: 		Policy (xi [bs, xi_dim]) -> (u [bs, u_dim]) -> (xi, [bs, xi_dim]), (ws [bs,])

	:return:
	"""

	xi_dim = xi0.shape[-1]

	if xi0.shape.ndims == 2:
		batch_shape = xi0.shape[0]
	else:
		assert batch_shape is not None, "If xi0 is not batch, batch_shape should be given"
		xi0 = xi0[None] * tf.ones(batch_shape)[:, None]

	xs = xi0[None]

	if isinstance(t0, int):
		i0 = tf.constant(t0)  # counter
	else:
		i0 = t0

	c = lambda i, xs, us, ws: tf.less(i, t0 + T)  # condition
	us = tf.zeros([1, batch_shape, u_dim])
	ws = tf.zeros([1, batch_shape])

	def pred(i, xs, us, ws):
		_us = pi(xs[-1], i)

		if isinstance(_us, tuple):
			next_us, next_ws_us = _us
		else:
			next_us, next_ws_us = _us, tf.zeros_like(_us[:, 0])

		try:
			_xs = f(xs[-1], next_us, i)
		except TypeError:
			_xs = f(xs[-1], next_us)

		if isinstance(_xs, tuple):
			next_xs, next_ws_xs = _xs
		else:
			next_xs, next_ws_xs = _xs, tf.zeros_like(_xs[:, 0])

		next_ws = next_ws_us + next_ws_us

		return tf.add(i, 1), tf.concat([xs, next_xs[None]], axis=0), \
			   tf.concat([us, next_us[None]], axis=0), tf.concat([ws, next_ws[None]],
																 axis=0)

	_, xs_pred, us_pred, ws_pred = tf.while_loop(
		c, pred, loop_vars=[i0, xs, us, ws], shape_invariants=
		[i0.get_shape(), tf.TensorShape([None, batch_shape, xi_dim]),
		 tf.TensorShape([None, batch_shape, u_dim]), tf.TensorShape([None, batch_shape])]
	)

	if time_first:
		return xs_pred, us_pred, ws_pred
	else:
		return tf.transpose(xs_pred, (1, 0, 2)), tf.transpose(us_pred,
															  (1, 0, 2)), tf.transpose(
			ws_pred, (1, 0))


def make_rollout_samples(xi0, f, pi, u_dim, T=100, batch_shape=None, t0=0,
						 time_first=False):
	"""

	:param xi0: 	Initial state [batch_shape, xi_dim] or [xi_dim]
	:param f: 		Dynamic model (xi [bs, xi_dim], u [bs, u_dim]) -> (xi [bs, xi_dim])
	:param pi: 		Policy (xi [bs, xi_dim]) -> (u [bs, u_dim])

	:return:
	"""

	xi_dim = xi0.shape[-1]

	if xi0.shape.ndims == 2:
		batch_shape = xi0.shape[0]
	else:
		assert batch_shape is not None, "If xi0 is not batch, batch_shape should be given"
		xi0 = xi0[None] * tf.ones(batch_shape)[:, None]

	xs = xi0[None]

	if isinstance(t0, int) or isinstance(t0, np.int64):
		i0 = tf.constant(t0)  # counter
	else:
		i0 = t0

	c = lambda i, xs, us: tf.less(i, t0 + T)  # condition
	us = tf.zeros([1, batch_shape, u_dim])

	def pred(i, xs, us):
		next_us = pi(xs[-1], i)
		try:
			next_xs = f(xs[-1], next_us, i)
		except TypeError:
			next_xs = f(xs[-1], next_us)

		return tf.add(i, 1), tf.concat([xs, next_xs[None]], axis=0), tf.concat(
			[us, next_us[None]], axis=0)

	_, xs_pred, us_pred = tf.while_loop(
		c, pred, loop_vars=[i0, xs, us], shape_invariants=
		[i0.get_shape(), tf.TensorShape([None, batch_shape, xi_dim]),
		 tf.TensorShape([None, batch_shape, u_dim])]
	)

	# us_pred = us_pred[1:]

	if time_first:
		return xs_pred, us_pred
	else:
		return tf.transpose(xs_pred, (1, 0, 2)), tf.transpose(us_pred, (1, 0, 2))


def make_multi_shooting_rollout_samples(
		xi0, f, pi, u_dim, T=100, batch_shape=None, t0=0, horizon=20, time_first=False,
		batch_shape_env=None, covs_diag_init=0.1, importance_sampling=False):
	idx_start = list(range(0, T, horizon))
	idx_end = idx_start[1:]

	if idx_start[-1] == T:
		idx_start = idx_start[:-1]
	else:
		idx_end += [T]

	idx_start = np.array(idx_start)
	idx_end = np.array(idx_end)
	horizons = idx_end - idx_start

	n_shoots = len(horizons)

	xi_dim = xi0.shape[-1] if tf.__version__[0] == '2' else xi0.shape[-1].value

	if batch_shape_env is None:
		xi0_loc = make_loc((n_shoots - 1, xi_dim), 0., 1.)
		xi0_covs = make_cov(
			xi_dim, covs_diag_init, param='diag', is_prec=False,
			batch_shape=(n_shoots - 1,))

		p_xi0 = MultivariateNormalFullCovariance(xi0_loc, xi0_covs)
		xi0s = [xi0] + tf.unstack(p_xi0.sample(batch_shape), axis=1)
	else:
		xi0_loc = make_loc((n_shoots - 1, batch_shape_env, xi_dim,), 0., 1.)
		xi0_covs = make_cov(xi_dim, covs_diag_init, param='diag', is_prec=False,
							batch_shape=(n_shoots - 1, batch_shape_env,))

		p_xi0 = MultivariateNormalFullCovariance(xi0_loc, xi0_covs)

		xi0s = [xi0] + tf.unstack(
			tf.reshape(
				tf.transpose(p_xi0.sample(batch_shape), (1, 2, 0, 3)),
				(n_shoots - 1, batch_shape_env * batch_shape, xi_dim)), axis=0)

	xs, us, ws = [], [], []

	for i in range(n_shoots):
		if importance_sampling:
			_xs, _us, _ws = make_rollout_samples_is(
				xi0s[i], f, pi, u_dim=u_dim, T=horizons[i] - 1, t0=idx_start[i],
				batch_shape=batch_shape, time_first=time_first)

			ws += [_ws]
		else:
			_xs, _us = make_rollout_samples(
				xi0s[i], f, pi, u_dim=u_dim, T=horizons[i] - 1, t0=idx_start[i],
				batch_shape=batch_shape, time_first=time_first)

		xs += [_xs]
		us += [_us]

	if importance_sampling:
		return xs, us, ws, [xi0_loc, xi0_covs], p_xi0
	else:
		return xs, us, [xi0_loc, xi0_covs], p_xi0

def make_rollout_autonomous_mvn(p_xi0, f, T=100, return_ds=True):
	"""

	:param p_xi0: 	(loc, cov) [xi_dim]
	:param f: 		Dynamic model ((loc, cov) xi [xi_dim) -> ((loc, cov) xi [xi_dim])
	:return:
	"""

	if isinstance(p_xi0, tuple):
		xi0_loc, xi0_cov = p_xi0
	else:
		xi0_loc, xi0_cov = p_xi0.loc, p_xi0.covariance()

	xi_dim = xi0_loc.shape[-1].value

	xs_loc = xi0_loc[None]
	xs_cov = xi0_cov[None]

	i0 = tf.constant(0)  # counter

	c = lambda i, xs_loc, xs_cov: tf.less(i, T)  # condition

	def pred(i, xs_loc, xs_cov):
		next_xs_loc, next_xs_cov = f((xs_loc[-1], xs_cov[-1]), i)
		return tf.add(i, 1), \
			   tf.concat([xs_loc, next_xs_loc[None]], axis=0), tf.concat(
			[xs_cov, next_xs_cov[None]], axis=0)

	_, xs_loc, xs_cov = tf.while_loop(
		c, pred, loop_vars=[i0, xs_loc, xs_cov], shape_invariants=
		[
			i0.get_shape(),
			tf.TensorShape([None, xi_dim]), tf.TensorShape([None, xi_dim, xi_dim])
		]
	)

	if return_ds:
		return MultivariateNormalFullCovariance(xs_loc, xs_cov)
	else:
		return xs_loc, xs_cov


def make_rollout_mvn(p_xi0, f, pi, u_dim=2, T=100, return_ds=True,
					 reg_u=None, reg_xi=None):
	"""

	:param p_xi0: 	(loc, cov) [xi_dim]
	:param f: 		Dynamic model ((loc, cov) xi [xi_dim], (loc, cov) u [u_dim]) -> ((loc, cov) xi [xi_dim])
	:param pi: 		Policy ((loc, cov) xi [xi_dim]) -> ((loc, cov) u [u_dim])
	:return:
	"""

	if isinstance(p_xi0, tuple):
		xi0_loc, xi0_cov = p_xi0
	else:
		xi0_loc, xi0_cov = p_xi0.loc, p_xi0.covariance()

	xi_dim = xi0_loc.shape[-1].value

	xs_loc = xi0_loc[None]
	xs_cov = xi0_cov[None]

	i0 = tf.constant(0)  # counter

	c = lambda i, xs_loc, xs_cov, us_loc, us_cov: tf.less(i, T)  # condition

	us_loc = tf.zeros([1, u_dim])
	us_cov = tf.zeros([1, u_dim, u_dim])

	def pred(i, xs_loc, xs_cov, us_loc, us_cov):
		next_us_loc, next_us_cov = pi((xs_loc[-1], xs_cov[-1]), i)
		next_xs_loc, next_xs_cov = f((xs_loc[-1], xs_cov[-1]), (next_us_loc, next_us_cov))
		return tf.add(i, 1), \
			   tf.concat([xs_loc, next_xs_loc[None]], axis=0), tf.concat(
			[xs_cov, next_xs_cov[None]], axis=0), \
			   tf.concat([us_loc, next_us_loc[None]], axis=0), tf.concat(
			[us_cov, next_us_cov[None]], axis=0)

	_, xs_loc, xs_cov, us_loc, us_cov = tf.while_loop(
		c, pred, loop_vars=[i0, xs_loc, xs_cov, us_loc, us_cov], shape_invariants=
		[
			i0.get_shape(),
			tf.TensorShape([None, xi_dim]), tf.TensorShape([None, xi_dim, xi_dim]),
			tf.TensorShape([None, u_dim]), tf.TensorShape([None, u_dim, u_dim])
		]
	)

	us_loc = us_loc[1:]
	us_cov = us_cov[1:]

	if reg_u:
		us_cov += reg_u ** 2 * tf.eye(u_dim)[None]

	if reg_xi:
		xs_cov += reg_xi ** 2 * tf.eye(xi_dim)[None]

	if return_ds:
		return MultivariateNormalFullCovariance(xs_loc, xs_cov), \
			   MultivariateNormalFullCovariance(us_loc, us_cov)
	else:
		return xs_loc, xs_cov, us_loc, us_cov
