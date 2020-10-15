import tensorflow as tf
from ..utils.tf_utils import batch_jacobians, matvec, matquad
from ..utils.param_utils import make_loc, make_cov
from tensorflow_probability import distributions as ds


class PoEPolicy(object):
	def __init__(self, product_size, experts_size, fs, js=None, *args, **kwargs):
		"""

		:param product_size  int
		:param experts_size  list
		:param fs:	list of policy
		:param fs:	list of transformations
		:param js:	list of jacobians, optional but will be slower if don't given
		"""

		assert isinstance(experts_size, list)

		self._product_size = product_size
		self._experts_size = experts_size


		self._fs = fs
		self._js = js

		self._n_experts = len(self._fs)

		if self._js is None:
			self._js = [None for i in range(self.n_experts)]

		for i in range(self.n_experts):
			if self._js[i] is None:
				self._js[i] = lambda x: tf.linalg.LinearOperatorFullMatrix(batch_jacobians(self._fs[i](x), x))

	@property
	def product_size(self):
		return self._product_size

	@property
	def experts_size(self):
		return self._experts_size

	@property
	def js(self):
		return self._js

	@property
	def fs(self):
		return self._fs

	@property
	def n_experts(self):
		return self._n_experts

	def jacobians(self, x):
		return [j(x) for j in self._js]

class ForcePoEPolicy(PoEPolicy):
	def __init__(self, pis, u_dim, *args, **kwargs):
		"""

		:param pis:    	list of policies [y, dy, t] -> u, cov_u
		:param u_dim:
		:param args:
		:param kwargs:
		"""
		self._pis = pis
		self._reg = 0.1

		self._u_dim = u_dim

		super(ForcePoEPolicy, self).__init__(*args, **kwargs)

	@property
	def reg(self):
		return self._reg

	@reg.setter
	def reg(self, value):
		self._reg = value

	@property
	def pis(self):
		return self._pis

	def density(self, xi, t=0):
		x, dx = xi[:, :self._u_dim], xi[:, self._u_dim:]

		ys = [f(x) for f in self.fs]  # transform state
		js = [j(x) for j in self.js]  # get jacobians
		dys = [j.matvec(dx) for j in js]   # get velocities in transformed space

		# "forces" in transformed space from the different policies
		fys_locs_covs = [self.pis[i](ys[i], dys[i], t) for i in range(self.n_experts)]

		# separate locs and covs
		fys_locs = [_y[0] for _y in fys_locs_covs]
		fys_covs = [_y[1] for _y in fys_locs_covs]

		# "forces" in original space
		fxs = [js[i].matvec(fys_locs[i], adjoint=True) for i in range(self.n_experts)]

		# covariances "forces" in original space
		fxs_covs = [matquad(js[i], fys_covs[i] , adjoint=True) for i in
					range(self.n_experts)]

		# precisions with regularization
		fxs_precs = [tf.linalg.inv(cov + self._reg ** 2 * tf.eye(self.experts_size[i])) for i, cov in
					 enumerate(fxs_covs)]

		# compute product of Gaussian policies
		precs = tf.reduce_sum(fxs_precs, axis=0)
		covs = tf.linalg.inv(precs)
		locs = [tf.linalg.LinearOperatorFullMatrix(fxs_precs[i]).matvec(fxs[i]) for i in
				range(self.n_experts)]
		locs = tf.linalg.LinearOperatorFullMatrix(covs).matvec(tf.reduce_sum(locs, axis=0))

		return ds.MultivariateNormalTriL(locs, tf.linalg.cholesky(covs))

	def sample(self, xi, t=0):
		return self.density(xi, t).sample()

class VelocityPoEPolicy(ForcePoEPolicy):
	def density(self, xi, t=0):
		ys = [f(xi) for f in self.fs]  # transform state
		js = [j(xi) for j in self.js]  # get jacobians

		# "velocities" in transformed space from the different policies
		fys_locs_covs = [self.pis[i](ys[i], t) for i in range(self.n_experts)]

		# separate locs and covs
		fys_locs = [_y[0] for _y in fys_locs_covs]
		fys_covs = [_y[1] for _y in fys_locs_covs]

		# precisions with regularization J^T Lambda
		fys_precs = [tf.linalg.inv(fys_covs[i] + self._reg ** 2 * tf.eye(self.experts_size[i]))
					  for i in range(self.n_experts)]

		fxs_eta = [tf.linalg.LinearOperatorFullMatrix(js[i].matmul(
			fys_precs[i], adjoint=True)).matvec(
			fys_locs[i]) for i in range(self.n_experts)]

		fxs_precs = [
			matquad(js[i], fys_precs[i]) for i in range(self.n_experts)
		]

		# compute product of Gaussian policies
		precs = tf.reduce_sum(fxs_precs, axis=0)

		covs = tf.linalg.inv(precs)

		etas = tf.reduce_sum(fxs_eta, axis=0)
		locs = tf.linalg.LinearOperatorFullMatrix(covs).matvec(etas)

		return ds.MultivariateNormalTriL(locs, tf.linalg.cholesky(covs))

class AccLQRPoEPolicy(PoEPolicy):
	def __init__(self, dt=0.01,
				 # S_inv_scale=None, S_param=None,
				 # Sv_inv_scale=1., Sv_param='iso',
				 # R_inv_scale=20., R_param='iso',
				 *args, **kwargs):
		"""
		$K_k = (R + B^T S_{K+1} B)^{-1} B^T S_{k+1} A$

		:param dt:
		:param S_inv_scale:			[list of float]
		:param S_param:				[list of string]
		:param Sv_inv_scale:  		[float]
			value function for velocity
		:param Sv_param:			[string]
		:param R_inv_scale:
		:param R_param:
		:param args:
		:param kwargs:
		"""
		self._dt = dt
		super(AccLQRPoEPolicy, self).__init__(*args, **kwargs)
		#
		# self._S = []
		# for i in range(self.n_experts):
		# 	self._S += [make_cov(
		# 		self.experts_size[i], 15., param='iso', is_prec=False,
		# 		batch_shape=(k_basis_cov,))]

		# S_param is None:

	def Ks(self, x, S, Sv, R):
		pass

	def As(self, x):
		As = [None for i in range(self.n_experts)]

		for i in range(self.n_experts):
			As[i] = tf.linalg.LinearOperatorFullMatrix(tf.concat([
				tf.concat([
					tf.eye(self.experts_size[i], batch_shape=(x.shape[0], )),
					self._js[i](x) * self._dt
				], axis=2),
				tf.concat([
					tf.zeros((x.shape[0], self.product_size, self.experts_size[i])),
					tf.eye(self.product_size, batch_shape=(x.shape[0], ))
				], axis=2),
			], axis=1))

		return As

	def Bs(self, x):
		Bs = [None for i in range(self.n_experts)]

		# for i in range(self.n_experts):
		# 	Bs[i] = tf.linalg.LinearOperatorFullMatrix(tf.concat([
		# 			0.5 * self._js[i](x) * self._dt ** 2,
		# 			tf.eye(self.product_size, batch_shape=(x.shape[0],)) * self._dt
		# 	], axis=1))

		for i in range(self.n_experts):
			Bs[i] = tf.linalg.LinearOperatorFullMatrix(
				0.5 * self._js[i](x) * self._dt ** 2)

		return Bs
