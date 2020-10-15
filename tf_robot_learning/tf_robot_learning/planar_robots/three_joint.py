import tensorflow as tf
from .robot import Robot
from tensorflow_probability import distributions as ds

pi = 3.14159

class ThreeJointRobot(Robot):
	def __init__(self, q=None, dq=None, ddq=None, ls=None, session=None):
		Robot.__init__(self)

		self._ls = tf.constant([0.24, 0.25, 0.25]) if ls is None else ls

		margin = 0.02
		self._joint_limits = tf.constant([[0. + margin, pi - margin],
										  [-pi + margin, pi - margin],
										  [-pi + margin, pi - margin]],
										 dtype=tf.float32)

		self._base_limits = tf.constant([[-1., 1.],
										  [-1., 1.]],
										 dtype=tf.float32)

		self._dof = 3

	def xs(self, q, x_base=None, angle=False):
		"""

		:param q:
		:param x_base: [2] or [batch_size, 2]
		:param angle:
		:return:
		"""
		if q.shape.ndims == 1:
			x = tf.cumsum([0,
						   self.ls[0] * tf.cos(q[0]),
						   self.ls[1] * tf.cos(q[0] + q[1]),
						   self.ls[2] * tf.cos(q[0] + q[1] + q[2])])
			y = tf.cumsum([0,
						   self.ls[0] * tf.sin(q[0]),
						   self.ls[1] * tf.sin(q[0] + q[1]),
						   self.ls[2] * tf.sin(q[0] + q[1] + q[2])])

			if x_base is not None:
				x += x_base[0]
				y += x_base[1]

			if angle:
				return tf.transpose(tf.stack([x, y, tf.cumsum(tf.concat([q, tf.zeros_like(q[0][None])], axis=0))]))
			else:
				return tf.transpose(tf.stack([x, y]))
		else:
			x = tf.cumsum([tf.zeros_like(q[:, 0]),
						   self.ls[None, 0] * tf.cos(q[:, 0]),
						   self.ls[None, 1] * tf.cos(q[:, 0] + q[:, 1]),
						   self.ls[None, 2] * tf.cos(q[:, 0] + q[:, 1] + q[:, 2])])

			y = tf.cumsum([tf.zeros_like(q[:, 0]),
						   self.ls[None, 0] * tf.sin(q[:, 0]),
						   self.ls[None, 1] * tf.sin(q[:, 0] + q[:, 1]),
						   self.ls[None, 2] * tf.sin(q[:, 0] + q[:, 1] + q[:, 2])])

			if x_base is not None:
				x += x_base[:, 0]
				y += x_base[:, 1]

			if angle:
				return tf.transpose(tf.stack([x, y, tf.cumsum(tf.concat([tf.transpose(q), tf.zeros_like(q[:, 0][None])], axis=0))]), (2, 1, 0))
			else:
				return tf.transpose(tf.stack([x, y]), (2, 1, 0))



	def base_limit_cost(self, x, std=0.1, base_limit=1.):
		return -ds.Normal(base_limit * self._base_limits[:, 0], std).log_cdf(x) - ds.Normal(-base_limit * self._base_limits[:, 1], std).log_cdf(-x)


	def J(self, q):
		q0 = q[:, 0]
		q01 = q[:, 0] + q[:, 1]
		q012 = q[:, 0] + q[:, 1] + q[:, 2]

		J = [[0. for i in range(3)] for j in range(2)]

		J[0][2] = self.ls[2] * -tf.sin(q012)
		J[1][2] = self.ls[2] * tf.cos(q012)
		J[0][1] = self.ls[1] * -tf.sin(q01) + J[0][2]
		J[1][1] = self.ls[1] * tf.cos(q01) + J[1][2]
		J[0][0] = self.ls[0] * -tf.sin(q0) + J[0][1]
		J[1][0] = self.ls[0] * tf.cos(q0) + J[1][1]

		arr = tf.stack(J)
		return tf.transpose(arr, (2, 0, 1))
