import tensorflow as tf
from tensorflow_probability import distributions as ds
from .rollout import make_rollout_mvn, make_rollout_samples, make_rollout_autonomous_mvn

def lqr_cost(Q, R, z=None, seq=None):
	def cost(x, u, Q, R, z=None, seq=None):
		"""

		:param x: [horizon, xi_dim] or [horizon, batch_size, xi_dim]
		:param u: [horizon, u_dim] or [horizon, batch_size, u_dim]
		:param Q:
		:param R:
		:param z:
		:param seq:
		:return:
		"""
		if x.shape.ndims == 2:
			if z is None:
				y = x
			elif z.shape.ndims == 2:
				if seq is not None:
					z = tf.gather(z, seq)
				y = z - x
			elif z.shape.ndims == 1:
				y = z[None] - x
			else:
				raise ValueError("Unknown target z")
		if x.shape.ndims == 3:
			if z is None:
				y = x
			elif z.shape.ndims == 2:
				if seq is not None:
					z = tf.gather(z, seq)
				y = z[:, None] - x
			elif z.shape.ndims == 1:
				y = z[None, None] - x
			else:
				raise ValueError("Unknown target z")

		if x.shape.ndims == 2:
			if Q.shape.ndims == 3:
				if seq is not None:
					Q = tf.gather(Q, seq)
				state_cost = tf.reduce_sum(y * tf.einsum('aij,aj->ai', Q, y))
			else:
				state_cost = tf.reduce_sum(y * tf.einsum('ij,aj->ai', Q, y))
		if x.shape.ndims == 3:
			if Q.shape.ndims == 3:
				if seq is not None:
					Q = tf.gather(Q, seq)
				state_cost = tf.reduce_sum(y * tf.einsum('aij,abj->abi', Q, y), axis=(0, 2))
			else:
				state_cost = tf.reduce_sum(y * tf.einsum('ij,abj->abi', Q, y), axis=(0, 2))

		if u.shape.ndims == 2:
			if R.shape.ndims == 3:
				control_cost = tf.reduce_sum(u * tf.einsum('aij,aj->ai', R, u))
			else:
				control_cost = tf.reduce_sum(u * tf.einsum('ij,aj->ai', R, u))
		if u.shape.ndims == 3:
			if R.shape.ndims == 3:
				control_cost = tf.reduce_sum(u * tf.einsum('aij,abj->abi', R, u), axis=(0, 2))
			else:
				control_cost = tf.reduce_sum(u * tf.einsum('ij,abj->abi', R, u), axis=(0, 2))

		return control_cost + state_cost

	return lambda x, u: cost(x, u, Q, R, z, seq)

def lqr(A, B, Q, R, z=None, horizon=None, seq=None):
	"""

	http://web.mst.edu/~bohner/papers/tlqtots.pdf


	:param A:   	[x_dim, x_dim]
	:param B: 		[x_dim, u_dim]
	:param Q:		[horizon, x_dim, x_dim] or [x_dim, x_dim]
 	:param R:		[horizon, u_dim, u_dim] or [u_dim, u_dim]
	:param z:		[horizon, x_dim] or [x_dim]
	:param seq:
	:param horizon:  int or tf.Tensor()
	:return:
	"""
	u_dim = B.shape[-1]
	x_dim = B.shape[0]

	if horizon is None:
		assert Q.shape.ndims == 3 and Q.shape[0] is not None, \
			"If horizon is not specified, Q should be of rank 3 with first dimension specified"

		horizon = Q.shape[0]

	if R.shape.ndims == 3:
		get_R = lambda i: R[i]
	else:
		get_R = lambda i: R

	if Q.shape.ndims == 3:
		if seq is None:
			get_Q = lambda i: Q[i]
		else:
			get_Q = lambda i: Q[seq[i]]
	else:
		get_Q = lambda i: Q

	if z is None:
		get_z = lambda i: tf.zeros(x_dim)
	elif z.shape.ndims == 2:
		if seq is None:
			get_z = lambda i: z[i]
		else:
			get_z = lambda i: z[seq[i]]
	else:
		get_z = lambda i: z

	### Init empty
	_Q = tf.zeros((0, u_dim, u_dim))
	_K = tf.zeros((0, u_dim, x_dim))
	_Kv = tf.zeros((0, u_dim, x_dim))

	S = get_Q(-1)[None]
	v = tf.matmul(get_Q(-1), get_z(-1)[:, None])[:, 0][None]

	i0 = tf.constant(0)  # counter

	c = lambda i, S, v, _Kv, _K, _Q: tf.less(i, horizon)  # condition


	def pred(i, S, v, _Kv, _K, _Q):
		prev_Q = tf.linalg.inv(get_R(-i) + tf.matmul(tf.matmul(B, S[0], transpose_a=True), B))
		prev_Kv = tf.matmul(prev_Q, B, transpose_b=True)  # already wrong
		prev_K = tf.matmul(tf.matmul(prev_Kv, S[0]), A)  # are already wrong

		AmBK = A - tf.matmul(B, prev_K)

		prev_S = tf.matmul(tf.matmul(A, S[0], transpose_a=True), AmBK) + get_Q(-i)
		prev_v = tf.matmul(AmBK, v[0][:, None], transpose_a=True)[:, 0] + \
				 tf.matmul(get_Q(-i), get_z(-i)[:, None])[:, 0]

		return tf.add(i, 1), tf.concat([prev_S[None], S], axis=0), tf.concat([prev_v[None], v],
																			 axis=0), \
			   tf.concat([prev_Kv[None], _Kv], axis=0), \
			   tf.concat([prev_K[None], _K], axis=0), tf.concat([prev_Q[None], _Q], axis=0)


	_, Ss, vs, _Kvs, _Ks, _Qs = tf.while_loop(
		c, pred, loop_vars=[i0, S, v, _Kv, _K, _Q], shape_invariants=
		[i0.get_shape(), tf.TensorShape([None, x_dim, x_dim]),
		 tf.TensorShape([None, x_dim]), tf.TensorShape([None, u_dim, x_dim]),
		 tf.TensorShape([None, u_dim, x_dim]), tf.TensorShape([None, u_dim, u_dim])
		 ]
	)

	return Ss, vs, _Kvs, _Ks, _Qs

class LQRPolicy(object):
	def __init__(self, A, B, Q, R, z=None, horizon=None, seq=None):
		self._Ss, self._vs, self._Kvs, self._Ks, self._Qs = lqr(
			A, B, Q, R, z=z, horizon=horizon, seq=seq)

		self.horizon = horizon
		self.A = A
		self.B = B


	def get_u_mvn(self, xi, i=0):
		xi_loc, xi_cov = xi

		u_loc = -tf.linalg.matvec(self._Ks[i], xi_loc) + \
			tf.linalg.matvec(self._Kvs[i], self._vs[i + 1])

		u_cov = self._Qs[i] + tf.linalg.matmul(
			tf.linalg.matmul(-self._Ks[i], xi_cov), -self._Ks[i], transpose_b=True)

		return u_loc, u_cov

	def f_mvn(self, xi, u, t=0):
		"""
		xi: [batch_size, xi_dim]
		u: [batch_size, u_dim]
		t : int
		return xi : [batch_size, xi_dim]
		"""
		u_loc, u_cov = u
		xi_loc, xi_cov = xi

		xi_n_loc = tf.linalg.matvec(self.A, xi_loc) + tf.linalg.matvec(self.B, u_loc)
		xi_n_cov = tf.linalg.matmul(tf.linalg.matmul(self.A, xi_cov), self.A, transpose_b=True) +\
			tf.linalg.matmul(tf.linalg.matmul(self.B, u_cov), self.B , transpose_b=True)

		return xi_n_loc, xi_n_cov

	def make_rollout_mvn(self, p_xi0, return_ds=True):

		def f_mvn(xi, i=0):
			xi_loc, xi_cov = xi

			xi_n_loc = tf.linalg.matvec(self.A, xi_loc) + tf.linalg.matvec(
				self.B, -tf.linalg.matvec(self._Ks[i], xi_loc) +
						tf.linalg.matvec(self._Kvs[i], self._vs[i + 1]))

			C = self.A - tf.linalg.matmul(self.B, self._Ks[i])
			xi_n_cov = tf.linalg.matmul(tf.linalg.matmul(C, xi_cov), C, transpose_b=True) +\
					   tf.linalg.matmul(tf.linalg.matmul(self.B, self._Qs[i]), self.B, transpose_b=True)

			return xi_n_loc, xi_n_cov

		return make_rollout_autonomous_mvn(
			p_xi0, f_mvn, T=self.horizon, return_ds=return_ds
		)

	def make_rollout_samples(self, p_xi0, n=10):
		xi0_samples = p_xi0.sample(n)

		def f(xi, u):
			return tf.linalg.LinearOperatorFullMatrix(self.A).matvec(xi) + \
				   tf.linalg.LinearOperatorFullMatrix(self.B).matvec(u)

		return make_rollout_samples(
			xi0_samples, f, self.get_u_samples, self.B.shape[-1], T=self.horizon,
			batch_shape=n
		)

	def get_u(self, xi, i=0):
		return -tf.einsum('ij,aj->ai', self._Ks[i], xi) + tf.einsum('ij,j->i', self._Kvs[i], self._vs[i+1])[None]

	def log_prob(self, us, xis, i=None):
		"""

		:param xis: [horizon + 1, batch_size, xi_dim]
		:param us: [horizon, batch_size, xi_dim]
		:param i:
		:return:
		"""
		if i is not None:
			raise NotImplementedError

		# compute u for each timestep and batch b, as size [horzon, batch_size, u_dim]
		u_locs = -tf.einsum('aij,abj->abi', self._Ks[:self.horizon], xis[:self.horizon]) + \
				 tf.einsum('aij,aj->ai', self._Kvs[:self.horizon], self._vs[1:self.horizon+1])[:, None]

		u_covs = self._Qs[:self.horizon][:, None]

		return ds.MultivariateNormalFullCovariance(loc=u_locs, covariance_matrix=u_covs).log_prob(us)

	def get_u_samples(self, xi, i=0):
		loc = -tf.einsum('ij,aj->ai', self._Ks[i], xi) + tf.einsum('ij,j->i', self._Kvs[i], self._vs[i+1])[None]
		cov = self._Qs[i]
		return ds.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=cov).sample(1)[0]

	def entropy(self):
		return tf.reduce_sum(ds.MultivariateNormalFullCovariance(
			covariance_matrix=self._Qs).entropy())
