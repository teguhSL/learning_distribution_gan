import tensorflow as tf

def log_normalize(x, axis=0):
	return x - tf.reduce_logsumexp(x, axis=axis, keepdims=True)

def _outer_squared_difference(x, y):
	"""Convenience function analogous to tf.squared_difference."""
	z = x - y
	return z[..., tf.newaxis, :] * z[..., tf.newaxis]

def reduce_cov(x, axis=0, weights=None):
	assert axis == 0, NotImplementedError

	if weights is None:
		return tf.reduce_mean(_outer_squared_difference(
				tf.reduce_mean(x, axis=0, keepdims=True), x), axis=0)
	else:
		return tf.reduce_sum(weights[:, None, None] * _outer_squared_difference(
				tf.reduce_mean(x, axis=0, keepdims=True), x), axis=0)




def block_diagonal(ms):
	"""
	Create a block diagonal matrix with a list of square matrices of same sizes

	:type ms: 		lisf of tf.Tensor	[..., n_dim, n_dim]
	:return:
	"""
	import numpy as np

	n_dims = np.array([m.shape[-1].value for m in ms])

	if np.sum((np.mean(n_dims) - n_dims) ** 2):  # check if not all same dims
		return block_diagonal_different_sizes(ms)

	s = ms[0].shape[-1].value
	z = ms[0].shape.ndims - 2  # batch dims
	n = len(ms)  # final size of matrix
	mat = []

	for i, m in enumerate(ms):
		nb, na = i * s, (n - i - 1) * s
		paddings = [[0, 0] for i in range(z)] + [[nb, na], [0, 0]]
		mat += [tf.pad(m, paddings=paddings)]

	return tf.concat(mat, -1)

def block_diagonal_different_sizes(ms):
	import numpy as np
	s = np.array([m.shape[-1].value for m in ms])

	cs = [0] + np.cumsum(s).tolist()
	z = ms[0].shape.ndims - 2  # batch dims
	mat = []

	for i, m in enumerate(ms):
		nb, na = cs[i], cs[-1] - cs[i] - s[i]
		paddings = [[0, 0] for i in range(z)] + [[nb, na], [0, 0]]
		mat += [tf.pad(m, paddings=paddings)]

	return tf.concat(mat, -1)

def matquad(lin_op, m, adjoint=False):
	"""
	A^T m A
	:param lin_op:
	:type lin_op: tf.linalg.LinearOperatorFullMatrix
	:param m:
	:param adjoint : A m A ^T
	:return:
	"""
	if isinstance(lin_op, tf.Tensor):
		lin_op = tf.linalg.LinearOperatorFullMatrix(lin_op)

	if adjoint:
		return lin_op.matmul(lin_op.matmul(m), adjoint_arg=True)

	return lin_op.matmul(lin_op.matmul(m, adjoint=True), adjoint=True, adjoint_arg=True)

def matvec(lin_op, v):
	"""
	A^T v A
	:param lin_op:
	:type lin_op: tf.linalg.LinearOperatorFullMatrix
	:param v:


	:return:
	"""
	return lin_op.matvec(lin_op.matvec(v, adjoint=True), adjoint=True)

def batch_jacobians(ys, xs):
	"""
	ys : [None, n_y] or [n_y]
	xs : [None, n_x] or [n_x]
	"""
	if ys.shape.ndims == 2:
		return tf.transpose(
			tf.stack([tf.gradients(ys[:, i], xs)[0] for i in range(ys.shape[-1].value)]),
			(1, 0, 2))
	elif ys.shape.ndims == 1:
		return tf.stack([tf.gradients(ys[i], xs)[0] for i in range(ys.shape[0].value)])
	else:
		raise NotImplementedError

def reduce_mvn_mm(locs=None, covs=None, h=None, axis=0):
	"""
	Perform moment matching

	:param locs: [..., n_dim]
	:param covs: [..., n_dim, n_dim]
	:param h: [...]
	:param axis:
	:return:
	"""

	if h is not None:
		# make h [..., 1], multiply and reduce
		loc = tf.reduce_sum(tf.expand_dims(h, -1) * locs, axis=axis)

		dlocs = locs - tf.expand_dims(loc, axis=axis)
		cov_locs = tf.matmul(tf.expand_dims(dlocs, axis=-1),
							 tf.expand_dims(dlocs, axis=-2))

		# make h [..., 1, 1]
		cov = tf.reduce_sum(
			tf.expand_dims(tf.expand_dims(h, -1), -1) * (covs + cov_locs), axis=axis)

	else:
		# make h [..., 1], multiply and reduce
		loc = tf.reduce_mean(locs, axis=axis)

		dlocs = locs - tf.expand_dims(loc, axis=axis)
		cov_locs = tf.matmul(tf.expand_dims(dlocs, axis=-1),
							 tf.expand_dims(dlocs, axis=-2))

		# make h [..., 1, 1]
		cov = tf.reduce_mean(covs + cov_locs, axis=axis)

	return loc, cov