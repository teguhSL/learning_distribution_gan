import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow_probability import distributions as ds
from ..utils.tf_utils import batch_jacobians

class PoE(ds.Distribution):
	def __init__(self, shape, experts, transfs, name='PoE', cost=None):
		"""

		:param shape:
		:param experts:
		:param transfs: 	a list of tensorflow function that goes from product space to expert space
			 or a function f(x: tensor, i: index of transform) -> f_i(x)

		:param cost: additional cost [batch_size, n_dim] -> [batch_size, ]
			a function f(x) ->
		"""

		self._product_shape = shape
		self._experts = experts
		self._transfs = transfs
		self._laplace = None
		self._samples_approx = None

		self._cost = cost


		self.stepsize = tf.Variable(0.01)
		self._name = name

		self._x_ph = tf1.placeholder(tf.float32, (None, shape[0]))
		self._y = None
		self._jac = None

	def f(self, x, return_jac=False, sess=None, feed_dict={}):
		dim1 = False


		if self._y is None:
			self._y = tf.concat(self.get_transformed(self._x_ph), axis=1)
		if return_jac and self._jac is None:
			self._jac = batch_jacobians(self._y, self._x_ph)

		if x.ndim == 1:
			x = x[None]
			dim1 = True

		if not return_jac:
			feed_dict[self._x_ph] = x
			_y = self._y.eval(feed_dict)
			if dim1: return _y[0]
			else: return _y
		else:
			if sess is None: sess = tf.get_default_session()

			feed_dict[self._x_ph] = x
			_y, _jac = sess.run([self._y, self._jac], feed_dict)

			if dim1: return _y[0], _jac[0]
			else: return _y, _jac

	def J(self, x, feed_dict={}):
		dim1 = False

		if self._y is None:
			self._y = tf.concat(self.get_transformed(self._x_ph), axis=1)
		if self._jac is None:
			self._jac = batch_jacobians(self._y, self._x_ph)

		if x.ndim == 1:
			x = x[None]
			dim1 = True

		feed_dict[self._x_ph] = x
		_jac = self._jac.eval(feed_dict)
		if dim1:
			return _jac[0]
		else:
			return _jac

	def get_loc_prec(self):
		raise NotImplementedError
		# return tf.concat([exp.mean() for exp in self.experts], axis=0),\
		# 	   block_diagonal_different_sizes(
		# 		   [tf.linalg.inv(exp.covariance()) for exp in self.experts])

	@property
	def product_shape(self):
		return self._product_shape

	@property
	def experts(self):
		return self._experts

	@property
	def transfs(self):
		return self._transfs


	def _experts_probs(self, x):
		probs = []

		for i, exp in enumerate(self.experts):
			if isinstance(self.transfs, list):
				if hasattr(exp, '_log_unnormalized_prob'):
					print('Using unnormalized prob for expert %d' % i)
					probs += [exp._log_unnormalized_prob(self.transfs[i](x))]
				else:
					probs += [exp.log_prob(self.transfs[i](x))]
			else:
				if hasattr(exp, '_log_unnormalized_prob'):
					probs += [exp._log_unnormalized_prob(self.transfs(x, i))]
				else:
					probs += [exp.log_prob(self.transfs(x, i))]

		return probs

	def _log_unnormalized_prob(self, x, wo_cost=False):
		if x.get_shape().ndims == 1:
			x = x[None]

		if wo_cost:
			return tf.reduce_sum(self._experts_probs(x), axis=0)
		else:
			cost = 0. if self._cost is None else self._cost(x)
			return tf.reduce_sum(self._experts_probs(x), axis=0) - cost

	@property
	def nb_experts(self):
		return len(self.experts)


	def get_ml_transformed(self, x):
		ys = self.get_transformed(x)

		return [[tf.reduce_mean(y, axis=0), reduce_cov(y, axis=0)] for y in ys]


	def get_transformed(self, x):
		return [f(x) for f in self.transfs]

