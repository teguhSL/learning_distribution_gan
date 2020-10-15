from tensorflow_probability import distributions as ds
import tensorflow as tf
from ..utils import tf_utils

tau = 6.283185307179586

class MultivariateNormalIso(ds.MultivariateNormalDiag):
	def __init__(self,
				 loc=None,
				 scale=None,
				 scale_identity_multiplier=None,
				 validate_args=False,
				 allow_nan_stats=True,
				 name="MultivariateNormalIso"):
		ds.MultivariateNormalDiag.__init__(
			self,
			loc=loc,
			scale_diag=tf.ones_like(loc) * scale,
			scale_identity_multiplier=scale_identity_multiplier,
			validate_args=validate_args,
			allow_nan_stats=allow_nan_stats,
			name=name
		)


class MultivariateNormalFullCovarianceML(ds.MultivariateNormalFullCovariance):
	def __init__(self,
				 loc=None,
				 covariance_matrix=None,
				 data=None,
				 trainable=False,
				 param='tril',
				 validate_args=False,
				 allow_nan_stats=True,
				 name="MultivariateNormalFullCovarianceML"):


		self._loc_var = loc

		if trainable:
			self._cov_expm = covariance_matrix
			if param == 'tril':
				self._cov_var = tf.linalg.matmul(covariance_matrix, covariance_matrix, transpose_b=True)
			elif param == 'diag':
				self._cov_var = tf.linalg.diag(tf.exp(covariance_matrix) ** 2)
			elif param == 'iso':
				self._cov_var = tf.linalg.diag(tf.exp(covariance_matrix) ** 2 * tf.ones_like(loc))
			else:
				self._cov_var = tf.linalg.expm(
					0.5 * (self._cov_expm+ tf.compat.v1.matrix_transpose(self._cov_expm)))
		else:
			self._cov_var = covariance_matrix

		if data is None:
			self._data_ml = tf.compat.v1.placeholder(loc.dtype, [None] + [s for s in loc.shape])
		else:
			self._data_ml = data

		self._n = loc.shape[-1]

		self._reg_diag = tf.compat.v1.placeholder(tf.float32, (self._n, ))

		self._loc_ml = tf.reduce_mean(self._data_ml, axis=0)
		self._cov_ml = tf_utils.reduce_cov(self._data_ml, axis=0) + tf.linalg.diag(
			self._reg_diag ** 2.)

		# self._cov_ml_sym = 0.5 * (self._cov_ml + tf.compat.v1.matrix_transpose(self._cov_ml))

		self._loc_assign = tf.compat.v1.placeholder(loc.dtype, loc.shape)
		self._cov_assign = tf.compat.v1.placeholder(covariance_matrix.dtype, self._cov_var.shape)

		if trainable:
			if param == 'tril':
				self._ml_op = [
					self._loc_var.assign(self._loc_ml),
					self._cov_expm.assign(tf.linalg.cholesky(self._cov_ml))
				]
				self._assign_op = [
					self._loc_var.assign(self._loc_assign),
					self._cov_expm.assign(tf.linalg.cholesky(self._cov_assign))
				]
			elif param == 'diag':
				self._ml_op = [
					self._loc_var.assign(self._loc_ml),
					self._cov_expm.assign(tf.math.log(tf.linalg.diag_part(self._cov_ml) ** 0.5))
				]
				self._assign_op = [
					self._loc_var.assign(self._loc_assign),
					self._cov_expm.assign(tf.math.log(tf.linalg.diag_part(self._cov_assign) ** 0.5))
				]
			elif param == 'iso':
				self._ml_op = [
					self._loc_var.assign(self._loc_ml),
					self._cov_expm.assign(tf.math.log(
						tf.reduce_mean(tf.linalg.diag_part(self._cov_ml), axis=-1, keepdims=True) ** 0.5))
				]
				self._assign_op = [
					self._loc_var.assign(self._loc_assign),
					self._cov_expm.assign(tf.math.log(
						tf.reduce_mean(tf.linalg.diag_part(self._cov_assign), axis=-1, keepdims=True) ** 0.5))
				]
			else:
				self._ml_op = [
					self._loc_var.assign(self._loc_ml),
					self._cov_expm.assign(
						tf.cast(tf.linalg.logm(tf.cast(self._cov_ml, tf.complex64)),
								tf.float32))
				]
				self._assign_op = [
					self._loc_var.assign(self._loc_assign),
					self._cov_expm.assign(
						tf.cast(tf.linalg.logm(tf.cast(self._cov_assign, tf.complex64)),
								tf.float32))
				]
		else:
			self._ml_op = [
				self._loc_var.assign(self._loc_ml),
				self._cov_var.assign(self._cov_ml)
			]
			self._assign_op = [
				self._loc_var.assign(self._loc_assign),
				self._cov_var.assign(self._cov_assign),
			]



		self._alpha = tf.Variable(0.9)

		_p_loc_ip_tmp = (1. - self._alpha) * self._loc_var +\
						self._alpha * self._loc_ml
		_p_cov_ip_tmp = (1. - self._alpha) * self._cov_var +\
						self._alpha * self._cov_ml

		_p_dloc_ipi = _p_loc_ip_tmp - self._loc_var
		_p_dloc_ipd = _p_loc_ip_tmp - self._loc_ml

		if loc.shape.ndims == 2:
				_p_dloc_cov = (1 - self._alpha) * _p_dloc_ipi[:, :, None] * _p_dloc_ipi[:, None] + \
							  self._alpha * _p_dloc_ipd[:, :, None] * _p_dloc_ipd[:, None]
		elif loc.shape.ndims == 3:
			_p_dloc_cov = (1 - self._alpha) * _p_dloc_ipi[:, :, :, None] * _p_dloc_ipi[:, :, None] + \
						  self._alpha * _p_dloc_ipd[:, :, :, None] * _p_dloc_ipd[:, :, None]
		elif loc.shape.ndims == 1:
			_p_dloc_cov = (1 - self._alpha) * _p_dloc_ipi[:, None] * _p_dloc_ipi[None] + \
						  self._alpha * _p_dloc_ipd[:, None] * _p_dloc_ipd[None]
		else:
			pass
			# raise NotImplementedError

		if trainable:
			if param == 'tril':
				self._online_ml_op = [
					self._loc_var.assign(_p_loc_ip_tmp),
					self._cov_expm.assign(tf.linalg.cholesky(_p_cov_ip_tmp + _p_dloc_cov))
				]
			elif param == 'diag':
				self._online_ml_op = [
					self._loc_var.assign(self._loc_assign),
					self._cov_expm.assign(tf.math.log(tf.linalg.diag_part(_p_cov_ip_tmp + _p_dloc_cov) ** 0.5))
				]
			elif param == 'iso':
				self._online_ml_op = [
					self._loc_var.assign(self._loc_assign),
					self._cov_expm.assign(tf.math.log(tf.reduce_mean(
						tf.linalg.diag_part(_p_cov_ip_tmp + _p_dloc_cov), axis=-1, keepdims=True) ** 0.5))
				]
			else:
				self._online_ml_op = [
					self._loc_var.assign(_p_loc_ip_tmp),
					self._cov_expm.assign(
						tf.cast(tf.linalg.logm(tf.cast(_p_cov_ip_tmp + _p_dloc_cov, tf.complex64)), tf.float32))
				]

		else:
			self._online_ml_op = [
				self._loc_var.assign(_p_loc_ip_tmp),
				self._cov_var.assign(_p_cov_ip_tmp + _p_dloc_cov)
			]

		ds.MultivariateNormalFullCovariance.__init__(
			self, loc=loc, covariance_matrix=self._cov_var,
			validate_args=validate_args, allow_nan_stats=allow_nan_stats,
			name=name
		)

	@property
	def params(self):
		return [self._loc_var, self._cov_expm]

	def covariance(self, name="covariance"):
		return tf.identity(self._cov_var, name=name)

	def precision(self, name="precision"):
		return tf.linalg.inv(self._cov_var, name=name)

	@property
	def data_ml(self):
		return self._data_ml

	@property
	def ml_op(self):
		return self._ml_op

	@property
	def online_ml_op(self):
		return self._online_ml_op

	@property
	def alpha(self):
		return self._alpha

	@property
	def reg_diag(self):
		return self._reg_diag

	def assign(self, loc, cov, sess=None):
		if sess is None:
			sess = tf.compat.v1.get_default_session()

		sess.run(self._assign_op, {self._loc_assign: loc, self._cov_assign: cov})

	def ml(self, x, reg_diag=None, sess=None):
		"""

		:param x:  [None] + shape of loc
		:param reg_diag: float or array [event_dim] std deviation
		:return:
		"""
		if sess is None:
			sess = tf.compat.v1.get_default_session()

		if reg_diag is None:
			reg_diag = [0.] * self._n
		elif isinstance(reg_diag, float):
			reg_diag = [reg_diag] * self._n

		sess.run([self.ml_op], {self.data_ml: x, self._reg_diag: reg_diag})

	def online_ml(self, x, alpha=0.9, reg_diag=None, sess=None):
		"""

		:param x:  [None] + shape of loc
		:return:
		"""
		if sess is None:
			sess = tf.compat.v1.get_default_session()

		if reg_diag is None:
			reg_diag = [0.] * self._n
		elif isinstance(reg_diag, float):
			reg_diag = [reg_diag] * self._n

		sess.run([self._online_ml_op],
				 {self.data_ml: x, self._alpha: alpha, self._reg_diag: reg_diag})



class MultivariateNormalFullPrecision(ds.MultivariateNormalFullCovariance):
	def __init__(self,
				 loc=None,
				 precision_matrix=None,
				 precision_matrix_log_det=None,
				 validate_args=False,
				 allow_nan_stats=True,
				 name="MultivariateNormalFullPrecision"):

		self.precision_operator = tf.linalg.LinearOperatorFullMatrix(
			precision_matrix
		)

		self._loc = loc
		self.prec = precision_matrix
		self.prec_log_det = precision_matrix_log_det

		ds.MultivariateNormalFullCovariance.__init__(
			self, loc=loc, covariance_matrix=tf.linalg.inv(precision_matrix),
			validate_args=validate_args, allow_nan_stats=allow_nan_stats,
			name=name
		)



	def _log_prob(self, y):

		y = tf.math.subtract(self.loc, y)
		x = self.precision_operator.matvec(y)
		_log_unnormalized = -tf.reduce_sum(tf.multiply(x, y), axis=-1)

		if self.prec_log_det is None:
			_m_log_normalization = tf.linalg.slogdet(self.prec)[1]
		else:
			_m_log_normalization = self.prec_log_det

		_m_log_normalization = _m_log_normalization + tf.log(tau) * self.event_shape[-1]

		return 0.5 * tf.add(_log_unnormalized, _m_log_normalization)