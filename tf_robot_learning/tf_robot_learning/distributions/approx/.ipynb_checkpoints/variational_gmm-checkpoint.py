import tensorflow as tf
import tensorflow.compat.v1 as tf1
from ...utils.tf_utils import log_normalize
from tensorflow_probability import distributions as _distributions


class VariationalGMM(object):
	def __init__(self, log_unnormalized_prob, gmm=None, k=10, loc=0., std=1., ndim=None, loc_tril=None,
				 samples=20, temp=1., cov_type='diag', loc_scale=1., priors_scale=1e1):
		"""

		:param log_unnormalized_prob:	Unnormalized log density to estimate
		:type log_unnormalized_prob: 	a tensorflow function that takes [batch_size, ndim]
			as input and returns [batch_size]
		:param gmm:
		:param k:		number of components for GMM approximation
		:param loc:		for initialization, mean
		:param std:		for initialization, standard deviation
		:param ndim:
		"""
		self.log_prob = log_unnormalized_prob
		self.ndim = ndim
		self.temp = temp

		if gmm is None:
			assert ndim is not None, "If no gmm is defined, should give the shape of x"

			if cov_type == 'diag':
				_log_priors_var = tf.Variable(1. / priors_scale * log_normalize(tf.ones(k)))
				log_priors = priors_scale * _log_priors_var

				if isinstance(loc, tf.Tensor) and loc.shape.ndims == 2:
					_locs_var = tf.Variable(1. / loc_scale * loc)
					locs = loc_scale * _locs_var
				else:
					_locs_var = tf.Variable(
						1. / loc_scale * tf.random.normal((k, ndim), loc, std))
					locs = loc_scale * _locs_var

				log_std_diags = tf.Variable(tf.log(std/k * tf.ones((k, ndim))))

				self._opt_params = [_log_priors_var, _locs_var, log_std_diags]

				gmm = _distributions.MixtureSameFamily(
					mixture_distribution=_distributions.Categorical(logits=log_priors),
					components_distribution=_distributions.MultivariateNormalDiag(
						loc=locs, scale_diag=tf.math.exp(log_std_diags)
					)
				)

			elif cov_type == 'full':
				_log_priors_var = tf.Variable(1./priors_scale * log_normalize(tf.ones(k)))
				log_priors = priors_scale * _log_priors_var

				if isinstance(loc, tf.Tensor) and loc.shape.ndims == 2:
					_locs_var = tf.Variable(1. / loc_scale * loc)
					locs = loc_scale * _locs_var
				else:
					_locs_var = tf.Variable(1./loc_scale * tf.random.normal((k, ndim), loc, std))
					locs = loc_scale * _locs_var


				loc_tril = loc_tril if loc_tril is not None else std/k
				# tril_cov = tf.Variable(loc_tril ** 2 * tf.eye(ndim, batch_shape=(k, )))

				tril_cov = tf.Variable(tf1.log(loc_tril) * tf.eye(ndim, batch_shape=(k, )))

				covariance = tf.linalg.expm(tril_cov + tf1.matrix_transpose(tril_cov))
				#
				self._opt_params = [_log_priors_var, _locs_var, tril_cov]

				gmm = _distributions.MixtureSameFamily(
					mixture_distribution=_distributions.Categorical(logits=log_priors),
					components_distribution=_distributions.MultivariateNormalFullCovariance(
						loc=locs, covariance_matrix=covariance
					)
				)

			else:
				raise ValueError("Unrecognized covariance type")

		self.k = k
		self.num_samples = samples
		self.gmm = gmm

	@property
	def sample_shape(self):
		return (self.num_samples, )

	@property
	def opt_params(self):
		"""
		Parameters to train
		:return:
		"""
		return self._opt_params

	@property
	def opt_params_wo_prior(self):
		return self._opt_params[1:]


	def mixture_elbo(self, *args):
		samples_conc = tf.reshape(
			tf.transpose(self.gmm.components_distribution.sample(self.sample_shape), perm=(1, 0, 2))
		, (-1, self.ndim)) # [k * nsamples, ndim]

		log_qs = tf.reshape(self.gmm.log_prob(samples_conc), (self.k, self.num_samples))
		log_ps = tf.reshape(self.temp * self.log_prob(samples_conc), (self.k, self.num_samples))

		component_elbos = tf.reduce_mean(log_ps-log_qs, axis=1)

		return tf.reduce_mean(component_elbos * tf.exp(log_normalize(self.gmm.mixture_distribution.logits)))


	def mixture_elbo_cst_prior(self, *args):
		samples = tf.reshape(self.gmm.components_distribution.sample(self.sample_shape), (-1, self.ndim))
		# samples = self.gmm.sample(self.sample_shape)

		log_ps = self.temp * self.log_prob(samples)
		log_qs = self.gmm.log_prob(samples)

		return tf.reduce_mean(log_ps - log_qs)


	@property
	def cost(self):
		return -self.mixture_elbo()

	@property
	def cost_cst_prior(self):
		return -self.mixture_elbo_cst_prior()