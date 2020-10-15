import tensorflow as tf
from .tf_utils import log_normalize

def has_parent_variable(tensor, variable):
	"""
	Check if tensor depends on variable
	:param tensor:
	:param variable:
	:return:
	"""
	var_list = len([var for var in tensor.op.values() if var == variable._variable])

	if var_list:
		return True

	for i in tensor.op.inputs:
		if has_parent_variable(i, variable):
			return True

	return False


def get_parent_variables(tensor, sess=None):
	"""
	Get all variables in the graph on which the tensor depends
	:param tensor:
	:param sess:
	:return:
	"""
	if sess is None:
		sess = tf.compat.v1.get_default_session()

	if tensor.__class__ == tf.Variable:
		return [tensor]

	return [v for v in sess.graph._collections['variables'] if
			has_parent_variable(tensor, v)]
#

class Param(tf.Tensor):
	def set_inverse_transform(self, f):
		self._inverse_transform = f

	def assign_op(self, value):
		return self._parent.assign(self._inverse_transform(value))

	def assign(self, value, sess=None):
		if sess is None:
			sess = tf.compat.v1.get_default_session()

		return sess.run(self.assign_op(value))

	@property
	def variable(self):
		return self._parent

	@variable.setter
	def variable(self, value):
		self._parent = value


def make_logits_from_value(priors):
	import numpy as np
	init_log = np.log(priors)

	v = tf.Variable(init_log, dtype=tf.float32)
	m = log_normalize(v, axis=-1)

	m.__class__ = Param
	m.variable = v

	return m

def make_loc_from_value(loc):
	v = tf.Variable(loc, dtype=tf.float32)
	m = tf.identity(v)

	m.__class__ = Param
	m.variable = v

	return m

def make_cov_from_value(cov, param='expm'):
	import numpy as np

	if param == 'expm':
		init_cov = tf.cast(
				tf.linalg.logm(cov.astype(np.complex64)),
			tf.float32)

		v = tf.Variable(init_cov)
		m = tf.linalg.expm(0.5 * (v + tf.compat.v1.matrix_transpose(v)))
	else:
		raise NotImplementedError

	m.__class__ = Param
	m.variable = v

	return m

def make_loc(shape, mean=0., scale=1.):
	v = tf.Variable(tf.random.normal(shape, mean, scale))
	m = tf.identity(v)

	m.__class__ = Param
	m.variable = v

	def inverse_transform(x):
		return tf.cast(x, dtype=tf.float32)

	m.set_inverse_transform(inverse_transform)
	return m

def make_rp(shape, mean=0.):
	v = tf.Variable(tf.zeros(shape) + tf.log(mean))
	m = tf.math.exp(v)

	m.__class__ = Param
	m.variable = v

	return m


def make_cov(k, scale=1., param='expm', batch_shape=(), is_prec=False, var=None):
	"""

	:param k: 		event_dim  spd will have shape (..., k, k)
	:type k : 		int
	:param scale: 	scale of the diagonal (std if covariance)
	:type scale:   	float, list of float, or tf.Tensor
	:param param: 	type of parametrization
	:type param:  	str in ['expm', 'tril', 'iso', 'diag']
	:param batch_shape:
	:param is_prec:	if True return a precision matrix whose std is scale
	:return:
	"""
	if isinstance(scale, float) and param is not 'iso':
		scale = scale * tf.ones(k)
	elif isinstance(scale, list):
		scale = tf.convert_to_tensor(scale)


	if param == 'expm':
		p = 2. if not is_prec else -2.
		v = tf.Variable(tf.eye(k, batch_shape=batch_shape) * tf.math.log(scale ** p)) if var is None else var
		m = tf.linalg.expm(0.5 * (v + tf.compat.v1.matrix_transpose(v)))

		m.__class__ = Param
		m.variable = v

		def inverse_transform(cov):
			import numpy as np
			return tf.cast(
				tf.linalg.logm(cov.astype(np.complex64)),
				tf.float32)

		m.set_inverse_transform(
			inverse_transform
		)

	elif param == 'tril':
		v = tf.Variable(tf.eye(k, batch_shape=batch_shape)) if var is None else var
		p = 1. if not is_prec else -1.
		m = v * (tf.linalg.diag(scale ** p) + tf.ones((k, k)) - tf.eye(k))
		m = tf.matmul(m, m, transpose_b=True)

		m.__class__ = Param
		m.variable = v

	elif param == 'iso':
		p = 2. if not is_prec else -2.
		v = tf.Variable(tf.ones(batch_shape + (1, 1)))
		m = v * tf.math.log(scale)
		m = tf.math.exp(m) ** p * tf.eye(k, batch_shape=batch_shape) if var is None else var

		m.__class__ = Param
		m.variable = v

	elif param == 'diag':
		p = 2. if not is_prec else -2.
		v = tf.Variable(tf.ones(batch_shape + (k,)) * tf.math.log(scale)) if var is None else var
		m = tf.compat.v1.matrix_diag(tf.math.exp(v) ** p)

		m.__class__ = Param
		m.variable = v

		def inverse_transform(cov):
			import numpy as np
			return 1./p * tf.log(tf.cast(tf.compat.v1.matrix_diag_part(cov), dtype=tf.float32))

		m.set_inverse_transform(
			inverse_transform
		)

	else:
		raise ValueError('param should be in [expm, tril, iso, diag]')



	return m
