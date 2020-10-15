import tensorflow as tf
from .robot import Robot

from tensorflow_probability import distributions as ds
from tensorflow_probability import math

pi = 3.14159

class TwoJointRobot(Robot):
	def __init__(self, ms=None, ls=None, dt=0.01, g=9.81):

		Robot.__init__(self)

		self._Js_com = None

		m1 = 5.
		m2 = 5.
		self._ms = tf.constant([m1,m2]) if ms is None else ms

		# I = tf.constant([
		# 		[1e-10,  1e-19, 1/12.],
		# 		[1e-10,  1e-10, 1/12.],
		# 	 ])

		I = tf.constant([
				[1e-2,  1e-2, 1/12.],
				[1e-2,  1e-2, 1/12.],
			 ])

		self._mass = tf.stack([
			tf.compat.v1.matrix_diag(
				tf.concat([
					tf.ones(3) * self._ms[0],
					self._ms[0] * I[0]
				], axis=0)),

			tf.compat.v1.matrix_diag(
				tf.concat([
					tf.ones(3) * self._ms[1],
					self._ms[1] * I[1]
				], axis=0))

		])


		L = 1.

		self._ls = tf.constant([L, L]) if ls is None else ls

		margin = 0.02

		self._joint_limits = tf.constant([
			[0. + margin, pi - margin],
			[-pi + margin, pi - margin],
		], dtype=tf.float32)

		self._dof = 2

		self._dt = dt
		self._g = g


	@property
	def mass(self):
		return self._mass

	def xs(self, q):
		if q.shape.ndims == 1:
			x = tf.cumsum([0,
						   self.ls[0] * tf.cos(q[0]),
						   self.ls[1] * tf.cos(q[0] + q[1])])
			y = tf.cumsum([0,
						   self.ls[0] * tf.sin(q[0]),
						   self.ls[1] * tf.sin(q[0] + q[1])])

			return tf.transpose(tf.stack([x, y]))

		else:
			x = tf.cumsum([tf.zeros_like(q[:, 0]),
						   self.ls[None, 0] * tf.cos(q[:, 0]),
						   self.ls[None, 1] * tf.cos(q[:, 0] + q[:, 1])])
			y = tf.cumsum([tf.zeros_like(q[:, 0]),
						   self.ls[None, 0] * tf.sin(q[:, 0]),
						   self.ls[None, 1] * tf.sin(q[:, 0] + q[: ,1])])

			return tf.transpose(tf.stack([x, y]), (2, 1, 0))

	def J(self, q):
		if q.shape.ndims == 1:
			J = [[0. for i in range(2)] for j in range(2)]
			J[0][1] = self.ls[1] * -tf.sin(q[0] + q[1])
			J[1][1] = self.ls[1] * tf.cos(q[0] + q[1])
			J[0][0] = self.ls[0] * -tf.sin(q[0]) + J[0][1]
			J[1][0] = self.ls[0] * tf.cos(q[0]) + J[1][1]

			arr = tf.stack(J)
			return tf.reshape(arr, (2, 2))
		else:
			J = [[0. for i in range(2)] for j in range(2)]
			J[0][1] = self.ls[1] * -tf.sin(q[:, 0] + q[:, 1])
			J[1][1] = self.ls[1] * tf.cos(q[:, 0] + q[:, 1])
			J[0][0] = self.ls[0] * -tf.sin(q[:, 0]) + J[0][1]
			J[1][0] = self.ls[0] * tf.cos(q[:, 0]) + J[1][1]

			arr = tf.stack(J)
			return tf.transpose(arr, (2, 0, 1))

	def f(self, x_t, u_t):
		"""
		To be used with tf_oc library
		State x_t :  [q, dq] in the shape [bs, 4]
		(x_t [bs, xi_dim], u_t [bs, u_dim]) -> (x_t+1 [bs, xi_dim])
		ddq = Mq_inv * (u - C*dq - g ), then integrate
		"""

		_q  = x_t[:,:2]
		_dq = x_t[:,2:4]


		_Mq_inv = self.Mq_inv(_q)
		_Cq     = self.Cq(_q,_dq)
		_gq     = self.gq(_q, self._g)
		_D      = self.D()


		M_ddq = -tf.einsum('aij,aj->ai',_Cq, _dq) - _gq + u_t - tf.einsum('ij,aj->ai',_D, _dq)
		ddq = tf.einsum('aij,aj->ai', _Mq_inv, M_ddq)
		dq  =  _dq + self._dt*ddq
		q   = _q + self._dt*dq + 0.5 * self._dt ** 2 * ddq

		return tf.concat([q, dq], axis=1)

	def Js_com(self, q):
		"""
		Recode Js_com with q [batch_size, 2]
		Can be found on
		http://www-lar.deis.unibo.it/people/cmelchiorri/Files_Robotica/FIR_05_Dynamics.pdf
		"""

		if q.shape.ndims == 1:
			_Js_com = [[[0. for i in range(2)] for j in range(6)] for k in range(2)]
			_Js_com[0][0][0] = self.ls[0] / 2. * -tf.sin(q[0])
			_Js_com[0][1][0] = self.ls[0] / 2. * tf.cos(q[0])
			_Js_com[0][5][0] = 1.0

			_Js_com[1][0][1] = self.ls[1] / 2. * -tf.sin(q[0] + q[1])
			_Js_com[1][1][1] = self.ls[1]/ 2. * tf.cos(q[0] + q[1])
			_Js_com[1][5][1] = 1.0
			_Js_com[1][0][0] = self.ls[0] * -tf.sin(q[0]) + _Js_com[1][0][1]
			_Js_com[1][1][0] = self.ls[0] * tf.cos(q[0]) + _Js_com[1][1][1]
			_Js_com[1][5][0] = 1.0

			return tf.stack(_Js_com)
		else:
			_Js_com = [[[tf.zeros_like(q[..., 0]) for i in range(2)] for j in range(6)] for k in range(2)]
			_Js_com[0][0][0] = self.ls[0] / 2. * -tf.sin(q[..., 0])
			_Js_com[0][1][0] = self.ls[0] / 2. * tf.cos(q[..., 0])
			_Js_com[0][5][0] = tf.ones_like(q[..., 0])

			_Js_com[1][0][1] = self.ls[1] / 2. * -tf.sin(q[..., 0] + q[..., 1])
			_Js_com[1][1][1] = self.ls[1]/ 2. * tf.cos(q[..., 0] + q[..., 1])
			_Js_com[1][5][1] = tf.ones_like(q[..., 0])
			_Js_com[1][0][0] = self.ls[0] * -tf.sin(q[..., 0]) + _Js_com[1][0][1]
			_Js_com[1][1][0] = self.ls[0] * tf.cos(q[..., 0]) + _Js_com[1][1][1]
			_Js_com[1][5][0] = tf.ones_like(q[..., 0])

			return tf.transpose(tf.stack(_Js_com), perm=(3, 0, 1, 2))

	def Mq(self, q):
		Js_com = self.Js_com(q)

		if q.shape.ndims == 1:
			_Mq = tf.matmul(Js_com[0], tf.matmul(self.mass[0], Js_com[0]), transpose_a=True) + \
			 tf.matmul(Js_com[1], tf.matmul(self.mass[1], Js_com[1]), transpose_a=True)
		else:
			_Mq = tf.einsum('aji,ajk->aik', Js_com[:, 0], tf.einsum('ij,ajk->aik',self.mass[0], Js_com[:, 0])) + \
				  tf.einsum('aji,ajk->aik', Js_com[:, 1], tf.einsum('ij,ajk->aik', self.mass[1], Js_com[:, 1]))
		return _Mq

	def Mq_inv(self,q):
		_Mq = self.Mq(q)
		return tf.linalg.inv(_Mq)

	def Cq(self,q,dq):
		if q.shape.ndims == 1:
			h = -self._ms[1]*self._ls[0]*(self._ls[1]*0.5)*tf.sin(q[1])
			_Cq = [[0. for i in range(2)] for j in range(2)]
			_Cq[0][0] = h*dq[1]
			_Cq[0][1] = h*(tf.reduce_sum(dq))
			_Cq[1][0] = -h*dq[0]
			return tf.stack(_Cq)

		else:
			h = -self._ms[1]*self._ls[0]*(self._ls[1]*0.5)*tf.sin(q[:, 1])
			_Cq = [[tf.zeros_like(q[:, 0]) for i in range(2)] for j in range(2)]
			_Cq[0][0] = h*dq[:, 1]
			_Cq[0][1] = h*(tf.reduce_sum(dq, axis=1))
			_Cq[1][0] = -h*dq[:, 0]

			return tf.transpose(tf.stack(_Cq), perm=(2, 0, 1))

	def gq(self, q, g):
		if q.shape.ndims==1:
			_gq = [0. for i in range(2)]

			a = (self._ms[0]*self._ls[0]/2 + self._ms[1]*self._ls[0])*g
			b = self._ms[1]*g*self._ls[1]*0.5

			_gq[0] = a*tf.cos(q[0]) + b*tf.cos(q[0] + q[1])
			_gq[1] = b*tf.cos(q[0] + q[1])
			return tf.transpose(tf.stack(_gq)[None])
		else:
			_gq = [tf.zeros_like(q[..., 0]) for i in range(2)]

			a = (self._ms[0]*self._ls[0]/2 + self._ms[1]*self._ls[0])*g
			b = self._ms[1]*g*self._ls[1]*0.5

			_gq[0] = a*tf.cos(q[..., 0]) + b*tf.cos(q[..., 0] + q[..., 1])
			_gq[1] = b*tf.cos(q[..., 0] + q[..., 1])

			return tf.transpose(tf.stack(_gq), perm=(1, 0))

	def D(self):
		return tf.eye(2) * .4





