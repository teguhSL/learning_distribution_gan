import tensorflow as tf
from ..nn import MLP
from ..utils.param_utils import make_cov

class Policy(object):
	def __init__(self, xi_dim, u_dim, *args, **kwargs):
		self._xi_dim = xi_dim
		self._u_dim = u_dim


	def pi(self, xi, t=0):
		"""

		:param xi: [batch_size, xi_dim]
		:return: u [batch_size, u_dim]
		"""

		raise NotImplementedError

	@property
	def u_dim(self):
		return self._u_dim

	@property
	def xi_dim(self):
		return self._xi_dim

class NNStochasticPolicy(Policy):
	def __init__(
			self, xi_dim, u_dim, noise_dim, n_hidden=[50, 50,],
			act_fct=tf.nn.tanh, noise_scale=1.):
		"""
		u = f(x) + noise

		:param xi_dim:
		:param u_dim:
		:param noise_dim:
		:param n_hidden:
		:param act_fct:
		"""

		Policy.__init__(self, xi_dim, u_dim)

		self._nn = MLP(
			n_input=xi_dim + noise_dim,
			n_output=u_dim,
			n_hidden=n_hidden,
			batch_size_svi=1,
			act_fct=act_fct
		)

		self._noise_dim = noise_dim
		self._noise_scale = noise_scale

	@property
	def params(self):
		return self._nn.vec_weights

	def pi(self, xi, t=0):
		return self._nn.pred(
			tf.concat([xi, tf.random.normal((xi.shape[0].value, self._noise_dim), 0., self._noise_scale) ], axis=1))

class NNStochasticFeedbackPolicy(Policy):
	def __init__(self, xi_dim, u_dim, noise_dim, A, B , n_hidden=[50, 50,], diag=False,
			act_fct=tf.nn.tanh, R_init=10, S_init=[0.02, 0.02, 0.5, 0.5], noise_scale=1.):
		"""

		u = K(x)(x_d(x) - x)

		:param xi_dim:
		:param u_dim:
		:param noise_dim:
		:param n_hidden:
		:param act_fct:
		"""
		Policy.__init__(self, xi_dim, u_dim)

		_n_output = xi_dim * xi_dim + xi_dim + 1 if not diag else xi_dim + xi_dim + 1

		self._nn = MLP(
			n_input=xi_dim + noise_dim,
			n_output=_n_output,
			n_hidden=n_hidden,
			batch_size_svi=1,
			act_fct=act_fct
		)

		self._diag = diag

		self._noise_scale = noise_scale

		self._S_init = S_init
		self._R_init = R_init

		self._A = A
		self._B = B

		self._noise_dim = noise_dim

	def pi(self, xi, t=0):
		pi_params = self._nn.pred(
			tf.concat([xi, tf.random.normal((xi.shape[0].value, self._noise_dim), 0., self._noise_scale)],
					  axis=1))

		xi_d, r, s = pi_params[:, :self.xi_dim], pi_params[:, self.xi_dim:self.xi_dim+1], \
					 pi_params[:, self.xi_dim+1:]

		R = tf.eye(self.u_dim, batch_shape=(xi.shape[0].value, )) * tf.math.exp(r)[:, None]

		if self._diag:
			S = tf.compat.v1.matrix_diag(tf.math.exp(s))
		else:
			s = tf.reshape(s, (-1, self.xi_dim, self.xi_dim))
			S = tf.linalg.expm(tf.compat.v1.matrix_transpose(s) + s)


		K = tf.matmul(
			tf.linalg.inv(R + self._B.matmul(
				self._B.matmul(S, adjoint=True), adjoint_arg=True, adjoint=True)),
			tf.linalg.LinearOperatorFullMatrix(self._B.matmul(S, adjoint=True)).matmul(self._A._matrix)
		)


		return tf.linalg.LinearOperatorFullMatrix(K).matvec(xi_d - xi)

	@property
	def params(self):
		return self._nn.vec_weights