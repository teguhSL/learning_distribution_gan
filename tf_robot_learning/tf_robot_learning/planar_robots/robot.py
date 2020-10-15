import tensorflow as tf

from tensorflow_probability import distributions as ds

pi = 3.14159

class Robot(object):
	def __init__(self):
		# robot description
		self._ls, self._ms, self._ins = [], [], []

		self._xs, self._J, self._Mn = None, None, None
		self._M_chol = None

		self._Mq = None

		self._mass = None
		self._joint_limits = None
		self._dof = None

		self._q_ev = None
		self._ev = {}

	@property
	def q_ev(self):
		if self._q_ev is None:
			self._q_ev = tf.compat.v1.placeholder(tf.float32, (None, self.dof))

		return self._q_ev

	def __getitem__(self, item):
		"""
		Get numpy version of the tensorflow function
		:param item:  ['xs', 'J', ...]
		:return:
		"""
		assert hasattr(self, item), 'Robot does not have this attribute'

		if not item in self._ev:
			self._ev[item] = getattr(self, item)(self.q_ev)

		sess = tf.compat.v1.get_default_session()

		return lambda x: sess.run(self._ev[item], {self._q_ev: x})

	@property
	def dof(self):
		return self._dof

	def gq(self, q, g):
		"""
		Gravity
		:param q:
		:param g:
		:return:
		"""
		raise NotImplementedError

	def Cq(self, q, dq):
		"""
		Coriolis
		:param q:
		:param dq:
		:return:
		"""
		raise NotImplementedError

	def Mq(self, q):
		"""
		Joint Inertia matrix
		:param q:
		:return:
		"""
		raise NotImplementedError

	def Mq_inv(self, q):
		"""
		Inverse of Joint Inertia matrix
		:param q:
		:return:
		"""
		raise NotImplementedError


	def Js_com(self, q):
		"""
		Jacobian of center of masses
		:param q:
		:return:
		"""
		raise NotImplementedError

	def f(self, x_t, u_t):
		"""
		Dynamic equation
		:param x_t:  [q, dq] in the shape [bs, 4]
		:param u_t: tau
		(x_t [bs, xi_dim], u_t [bs, u_dim]) -> (x_t+1 [bs, xi_dim])
		:return:
		"""
		raise NotImplementedError

	def joint_limit_cost(self, q, std=0.1):
		return -ds.Normal(self._joint_limits[:, 0], std).log_cdf(q) - \
			   ds.Normal(-self._joint_limits[:, 1], std).log_cdf(-q)

	@property
	def mass(self):
		"""
		Mass matrix of each segment
		:return:
		"""
		raise NotImplementedError

	def J(self, q):
		raise NotImplementedError

	def Mn(self, q):
		"""
		Manipulabililty
		:param q:
		:return:
		"""
		return tf.matmul(self.J(q), self.J(q), transpose_b=True)


	def xs(self, q):
		raise NotImplementedError

	@property
	def ms(self):
		"""
		Mass of each segment
		:return:
		"""
		return self._ms

	@property
	def ins(self):
		"""
		Inertia moment of each segment
		"""
		return self._ins

	@property
	def ls(self):
		return self._ls

	def segment_samples(self, q, nsamples_segment=10, noise_scale=None):

		if q.shape.ndims == 2:
			segments = self.xs(q)  # batch_shape, n_points, 2

			n_segments = segments.shape[1].value - 1
			samples = []

			for i in range(n_segments):
				u = tf.random_uniform((nsamples_segment,))

				# linear interpolations between end-points of segments
				#  batch_shape, nsamples_segment, 2
				samples += [u[None, :, None] * segments[:, i][:, None]
							+ (1. - u[None, :, None]) * segments[:, i + 1][:, None]]

				if noise_scale is not None:
					samples[-1] += tf.random_normal(samples[-1].shape, 0., noise_scale)

			return tf.concat(samples, axis=1)

		else:
			raise NotImplementedError

	def min_sq_dist_from_point(self, q, x, **kwargs):
		"""

		:param q: [batch_shape, 2]
		:param x: [2, ]
		:return:
		"""
		if q.shape.ndims == 2:
			samples = self.segment_samples(q, **kwargs)  # batch_shape, nsamples, 2

			dist = tf.reduce_sum((samples - x[None, None]) ** 2, axis=2)

			return tf.reduce_min(dist, axis=1)

	def plot(self, qs, ax=None, color='k', xlim=None, ylim=None, feed_dict={},
					ee_kwargs=None,
					dx=0.02, dy=0.02, fontsize=10, cmap=None, text=True, bicolor=None,
					x_base=None, **kwargs):
		import numpy as np
		import matplotlib.pyplot as plt

		qs = np.array(qs).astype(np.float32)

		if qs.ndim == 1:
			qs = qs[None]

		if x_base is None:
			xs = self['xs'](qs)
		else:
			raise NotImplementedError

		if text:
			for i in range(xs.shape[1] - 1):
				plt.annotate(r'$q_%d$' % i,
							 (xs[0, i, 0], xs[0, i, 1]),
							 (xs[0, i, 0] + dx, xs[0, i, 1] + dy), fontsize=fontsize)

		if hasattr(self, '_arms'):
			n_joint_arms = xs.shape[1] / 2
			for x in xs[:, :2]:
				plot = ax if ax is not None else plt
				plot.plot(x[:, 0], x[:, 1], marker='o', color='k', lw=10, mfc='w',
						  solid_capstyle='round',
						  **kwargs)

			for x in xs[:, 1:n_joint_arms]:
				plot = ax if ax is not None else plt
				plot.plot(x[:, 0], x[:, 1], marker='o', color=color, lw=10, mfc='w',
						  solid_capstyle='round',
						  **kwargs)

			for x in xs[:, n_joint_arms + 1:]:
				plot = ax if ax is not None else plt
				bicolor = color if bicolor is None else bicolor
				plot.plot(x[:, 0], x[:, 1], marker='o', color=bicolor, lw=10, mfc='w',
						  solid_capstyle='round',
						  **kwargs)

		else:
			alpha = kwargs.pop('alpha', 1.)
			plot = ax if ax is not None else plt
			plot.plot([], [], marker='o', color=color, lw=10, mfc='w',
					  solid_capstyle='round', alpha=1.,
					  **kwargs)

			kwargs.pop('label', None)
			kwargs['alpha'] = alpha

			for i, x in enumerate(xs):
				c = color if cmap is None else cmap(float(i) / (len(xs)))

				plot = ax if ax is not None else plt
				plot.plot(x[:, 0], x[:, 1], marker='o', color=c, lw=10, mfc='w',
						  solid_capstyle='round',
						  **kwargs)

				kwargs.pop('label', None)

			if ee_kwargs is not None:
				ee_kwargs['marker'] = ee_kwargs.pop('marker', 'x')
				ee_kwargs['ls'] = ' '
				plot = ax if ax is not None else plt
				plot.plot(xs[:, -1, 0], xs[:, -1, 1], **ee_kwargs)

		if ax is None:
			plt.axes().set_aspect('equal')
		else:
			ax.set_aspect('equal')

		if xlim is not None: plt.xlim(xlim)
		if ylim is not None: plt.ylim(ylim)
