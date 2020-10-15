from tensorflow_probability import distributions as ds
import tensorflow as tf
from ...utils import tf_utils, param
try:
	from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
except:
	print("Sklearn not installed, some features might not be usable")

import numpy as np

class EmptyMixture(object):
	def __init__(self):
		pass

def filter_unused(mixture, default_cov=None):
	idx = mixture.degrees_of_freedom_ >= float(
		mixture.degrees_of_freedom_prior_) + 0.5

	_mixture = EmptyMixture()
	_mixture.means_ = mixture.means_[idx]
	_mixture.covariances_ = mixture.covariances_[idx]
	_mixture.weights_ = mixture.weights_[idx]

	_mixture.means_  = np.concatenate([
		_mixture.means_, mixture.mean_prior_[None]], axis=0)

	_mixture.weights_ = np.concatenate([
		_mixture.weights_/np.sum(_mixture.weights_, keepdims=True) * 0.99, 0.01 * np.ones(1)], axis=0)

	if default_cov is None:
		default_cov = np.cov(mixture.means_.T) * np.eye(mixture.means_.shape[-1])

	_mixture.covariances_ = np.concatenate([
		_mixture.covariances_, default_cov[None]
	], axis=0)

	return _mixture

class GaussianMixtureModelFromSK(ds.MixtureSameFamily):
	def __init__(self, mixture, marginal_slice=None, bayesian=False):

		self._logits = param.make_logits_from_value(mixture.weights_)


		if bayesian:
			mixture = filter_unused(mixture)

		if marginal_slice is not None:
			self._locs = param.make_loc_from_value(
				mixture.means_[:, marginal_slice])
			self._covs = param.make_cov_from_value(
				mixture.covariances_[:, marginal_slice, marginal_slice])
		else:
			self._locs = param.make_loc_from_value(mixture.means_)
			self._covs = param.make_cov_from_value(mixture.covariances_)

		self._priors = tf.math.exp(self._logits)

		ds.MixtureSameFamily.__init__(
			self,
			mixture_distribution=ds.Categorical(probs=self._priors),
			components_distribution=ds.MultivariateNormalFullCovariance(
				loc=self._locs, covariance_matrix=self._covs
			)
		)

	@property
	def variables(self):
		return [self._logits.variable, self._locs.variable, self._covs.variable]

class GaussianMixtureModelML(ds.MixtureSameFamily):
	def __init__(self, priors, locs, covs, init_params='kmeans', warm_start=True,
				 reg_cov=1e-6, bayesian=False):

		self._priors = priors
		self._locs = locs
		self._covs = covs

		self._covs_par = covs

		if tf.__version__[0] == '1':
			self._k = priors.shape[0].value
			self._n = locs.shape[-1].value
		else:
			self._k = priors.shape[0]
			self._n = locs.shape[-1]

		m = BayesianGaussianMixture if bayesian else GaussianMixture
		self._sk_gmm = m(
			self._k, 'full', n_init=1, init_params=init_params, warm_start=warm_start, reg_covar=reg_cov)

		self._priors_ml = tf.compat.v1.placeholder(tf.float32, (self._k, ))
		self._locs_ml = tf.compat.v1.placeholder(tf.float32, (self._k, self._n))
		self._covs_ml = tf.compat.v1.placeholder(tf.float32, (self._k, self._n, self._n))

		self._ml_assign_op = [
			self._priors.assign(self._priors_ml),
			self._locs.assign(self._locs_ml),
			self._covs.assign(self._covs_ml)
		]

		ds.MixtureSameFamily.__init__(
			self,
			mixture_distribution=ds.Categorical(probs=self._priors),
			components_distribution=ds.MultivariateNormalFullCovariance(
				loc=self._locs, covariance_matrix=self._covs
			)
		)

		self._init = False

	def conditional_distribution(self, x, slice_in, slice_out):
		marginal_in = ds.MixtureSameFamily(
			mixture_distribution=ds.Categorical(probs=self._priors),
			components_distribution=ds.MultivariateNormalFullCovariance(
				loc=self._locs[:, slice_in], covariance_matrix=self._covs[:, slice_in, slice_in]
			))

		p_k_in = ds.Categorical(logits=tf_utils.log_normalize(
			marginal_in.components_distribution.log_prob(x[:, None])  +
			marginal_in.mixture_distribution.logits[None], axis=1))

		sigma_in_out = self._covs[:, slice_in, slice_out]
		inv_sigma_in_in = tf.linalg.inv(self._covs[:, slice_in, slice_in])
		inv_sigma_out_in = tf.matmul(sigma_in_out, inv_sigma_in_in, transpose_a=True)

		A = inv_sigma_out_in
		b = self._locs[:, slice_out] - tf.matmul(
			inv_sigma_out_in, self._locs[:, slice_in, None])[:, :, 0]

		cov_est = (self._covs[:, slice_out, slice_out] - tf.matmul(
			inv_sigma_out_in, sigma_in_out))

		ys = tf.einsum('aij,bj->abi', A, x) + b[:, None]

		p_out_in_k = ds.MultivariateNormalFullCovariance(
			tf.transpose(ys, perm=(1, 0, 2)), cov_est)

		return ds.MixtureSameFamily(
			mixture_distribution=p_k_in,
			components_distribution=p_out_in_k
		)

	def ml(self, x, sess=None, alpha=1., warm_start=True, n_init=1, reg_diag=None, max_iter=100):
		if not self._init: alpha = 1.
		self._init = True

		if alpha != 1.:
			prev_x = self._sk_gmm.sample(int(x.shape[0]*(1.-alpha)))[0]
			x = np.concatenate([x, prev_x], axis=0)

		if sess is None: sess = tf.compat.v1.get_default_session()

		self._sk_gmm.warm_start = warm_start
		self._sk_gmm.n_init = n_init
		self._sk_gmm.max_iter = max_iter
		self._sk_gmm.fit(x)

		_covs = self._sk_gmm.covariances_

		if reg_diag is not None:
			_covs += np.diag(reg_diag)[None] ** 2

		feed_dict = {
			self._priors_ml: self._sk_gmm.weights_,
			self._locs_ml: self._sk_gmm.means_,
			self._covs_ml: _covs,
		}

		sess.run(self._ml_assign_op, feed_dict)


