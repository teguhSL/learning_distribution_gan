import tensorflow as tf
from tensorflow_probability import distributions as ds
from ...utils.tf_utils import log_normalize
from ...utils import param

class Gate(object):
	def __init__(self):
		pass

	@property
	def opt_params(self):
		raise NotImplementedError

	def conditional_mixture_distribution(self, x):
		"""
		x : [batch_shape, dim]
		return [batch_shape, nb_experts]
		"""
		raise NotImplementedError


class Experts(object):
	def __init__(self):
		pass

	@property
	def opt_params(self):
		raise NotImplementedError

	@property
	def nb_experts(self):
		raise NotImplementedError

	@property
	def nb_dim(self):
		raise NotImplementedError


	def conditional_components_distribution(self, x):
		"""
		x : [batch_shape, dim]
		return distribution([batch_shape, nb_experts, dim_out])
		"""
		raise NotImplementedError


class LinearMVNExperts(Experts):
	def __init__(self, A, b, cov_tril):
		self._A = A
		self._b = b
		self._cov_tril = cov_tril
		self._covs = tf.linalg.LinearOperatorFullMatrix(
			self._cov_tril).matmul(self._cov_tril, adjoint_arg=True)

	@property
	def nb_dim(self):
		return self._b.shape[-1].value

	@property
	def nb_experts(self):
		return self._b.shape[0].value

	def conditional_components_distribution(self, x):
		ys = tf.einsum('aij,bj->abi', self._A, x) + self._b[:, None]

		return ds.MultivariateNormalTriL(
			tf.transpose(ys, perm=(1, 0, 2)),  # One for each component.
			self._cov_tril)

	@property
	def opt_params(self):
		return [self._A, self._b, self._cov_tril]

class MixtureGate(Gate):
	def __init__(self, mixture):
		self._mixture = mixture

		Gate.__init__(self)

	@property
	def opt_params(self):
		return self._mixture.opt_params

	@property
	def mixture(self):
		return self._mixture

	def conditional_mixture_distribution(self, x):
		# def logregexp(x, reg=1e-5, axis=0):
		# 	return [x, reg * tf.ones_like(x)]

		return ds.Categorical(logits=log_normalize(
			self._mixture.components_distribution.log_prob(x[:, None])  +
			self._mixture.mixture_distribution.logits[None], axis=1))


class MoE(object):
	def __init__(self, gate, experts, is_function=None):
		"""
		:type gate : Gate
		:type experts : Experts
		"""
		self._gate = gate
		self._experts = experts

		self._is_function = is_function

	@property
	def opt_params(self):
		return self._gate.opt_params + self._experts.opt_params

	@property
	def nb_experts(self):
		return self._experts.nb_experts

	def conditional_distribution(self, x):
		return ds.MixtureSameFamily(
			mixture_distribution=self._gate.conditional_mixture_distribution(x),
			components_distribution=self._experts.conditional_components_distribution(x)
		)


	def sample_is(self, x, n=1):
		mixture_distribution, mixture_components = \
			self._gate.conditional_mixture_distribution(x),\
			self._experts.conditional_components_distribution(x)

		y = mixture_components.sample(n)
		# npdt = y.dtype.as_numpy_dtype

		is_logits = self._is_function(mixture_distribution.logits)
		is_mixture_distribution = ds.Categorical(logits=is_logits)
		idx = is_mixture_distribution.sample(n)

		# TODO check if we should not renormalize mixture.logits - tf.stop_...

		weights = tf.batch_gather(mixture_distribution.logits - tf.stop_gradient(is_logits),
								  tf.transpose(idx))
		# TODO check axis
		# weights = tf.batch_gather(
		# 	log_normalize(mixture_distribution.logits - tf.stop_gradient(is_logits), axis=1),
		# 						  tf.transpose(idx))

		if n == 1:
			return tf.batch_gather(y, idx[:, :, None])[0, :, 0], tf.transpose(weights)[0]
		else:
			return tf.batch_gather(y, idx[:, :, None])[:, :, 0], tf.transpose(weights)

	@property
	def gate(self):
		return self._gate

	@property
	def experts(self):
		return self._experts

from .gmm_ml import GaussianMixtureModelFromSK, filter_unused
import numpy as np

class MoEFromSkMixture(MoE):
	def __init__(
			self, mixture, slice_in, slice_out, bayesian=False, default_cov=None,
			*args, **kwargs
	):
		if bayesian:
			mixture = filter_unused(mixture, default_cov=default_cov)

		gate = MixtureGate(GaussianMixtureModelFromSK(
			mixture=mixture, marginal_slice=slice_in, bayesian=False))

		sigma_in_out = mixture.covariances_[:, slice_in, slice_out]

		inv_sigma_in_in = np.linalg.inv(
			mixture.covariances_[:, slice_in, slice_in])
		inv_sigma_out_in = np.einsum(
			'aji,ajk->aik', sigma_in_out, inv_sigma_in_in)

		As = inv_sigma_out_in
		bs = mixture.means_[:, slice_out] - np.matmul(
			inv_sigma_out_in, mixture.means_[:, slice_in, None])[
								   :, :, 0]

		sigma_est = (mixture.covariances_[:, slice_out, slice_out] - np.matmul(
			inv_sigma_out_in, sigma_in_out))

		As = param.make_loc_from_value(As)
		bs = param.make_loc_from_value(bs)
		cov_tril = param.make_cov_from_value(np.linalg.cholesky(sigma_est))

		experts = LinearMVNExperts(
			A=As, b=bs, cov_tril=cov_tril
		)

		MoE.__init__(self, gate, experts, *args, **kwargs)

	@property
	def variables(self):
		return [self._experts._A.variable, self._experts._b.variable,
				self._experts._cov_tril.variable] + self._gate.mixture.variables