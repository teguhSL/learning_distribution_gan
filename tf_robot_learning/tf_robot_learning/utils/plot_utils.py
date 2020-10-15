import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
plt.style.use('ggplot')

class PolicyPlot(object):
	def __init__(self, pi, nb_sub=20):

		self._x = tf.placeholder(tf.float32, (nb_sub**2, 2))
		self._u = tf.transpose(pi(self._x))
		self._nb_sub = nb_sub
	def plot(self, ax=None, xlim=[-1, 1], ylim=[-1, 1], scale=0.01,
							name=None, equal=False, feed_dict={}, sess=None, **kwargs):
		"""
		Plot a dynamical system dx = f(x)
		:param f: 		a function that takes as input x as [N,2] and return dx [N, 2]
		:param nb_sub:
		:param ax0:
		:param xlim:
		:param ylim:
		:param scale:
		:param kwargs:
		:return:
		"""

		if sess is None:
			sess = tf.compat.v1.get_default_session()

		Y, X = np.mgrid[
			   ylim[0]:ylim[1]:complex(self._nb_sub),
			   xlim[0]:xlim[1]:complex(self._nb_sub)]
		mesh_data = np.vstack([X.ravel(), Y.ravel()])

		feed_dict[self._x] = mesh_data.T
		field = sess.run(self._u, feed_dict)

		U = field[0]
		V = field[1]
		U = U.reshape(self._nb_sub, self._nb_sub)
		V = V.reshape(self._nb_sub, self._nb_sub)
		speed = np.sqrt(U * U + V * V)

		if name is not None:
			plt.suptitle(name)

		if ax is not None:
			strm = ax.streamplot(X, Y, U, V, linewidth=scale * speed, **kwargs)
			ax.set_xlim(xlim)
			ax.set_ylim(ylim)

			if equal:
				ax.set_aspect('equal')

		else:
			strm = plt.streamplot(X, Y, U, V, linewidth=scale * speed, **kwargs)
			plt.xlim(xlim)
			plt.ylim(ylim)

			if equal:
				plt.axes().set_aspect('equal')

		return [strm]



class MVNPlot(object):
	def __init__(self, ds, nb_segments=20):

		from ..distributions import GaussianMixtureModelML, GaussianMixtureModelFromSK
		self._ds = ds
		if isinstance(ds, GaussianMixtureModelML) or isinstance(ds, GaussianMixtureModelFromSK):
			self._loc = ds.components_distribution.loc
			self._cov = ds.components_distribution.covariance()
		else:
			self._cov = ds.covariance()
			self._loc = ds.loc

		self._t = np.linspace(-np.pi, np.pi, nb_segments)

	def plot(self, *args, **kwargs):
		return self.plot_gmm(*args, **kwargs)


	def plot_gmm(self, dim=None, color=[1, 0, 0], alpha=0.5, linewidth=1,
				 markersize=6, batch_idx=0,
				 ax=None, empty=False, edgecolor=None, edgealpha=None,
				 border=False, nb=1, center=True, zorder=20, equal=True, sess=None,
				 feed_dict={}, axis=0):

		if sess is None:
			sess = tf.compat.v1.get_default_session()

		loc, cov = sess.run([self._loc, self._cov], feed_dict)


		if loc.ndim == 3:
			loc = loc[batch_idx] if axis==0 else loc[:, batch_idx]
		if cov.ndim == 4:
			cov = cov[batch_idx] if axis==0 else cov[:, batch_idx]

		if loc.ndim == 1:
			loc = loc[None]
		if cov.ndim == 2:
			cov = cov[None]

		nb_states = loc.shape[0]

		if dim:
			loc = loc[:, dim]
			cov = cov[np.ix_(range(cov.shape[0]), dim, dim)] if isinstance(dim,
																		   list) else cov[
																					  :,
																					  dim,
																					  dim]
		if not isinstance(color, list) and not isinstance(color, np.ndarray):
			color = [color] * nb_states
		elif not isinstance(color[0], str) and not isinstance(color, np.ndarray):
			color = [color] * nb_states

		if not isinstance(alpha, np.ndarray):
			alpha = [alpha] * nb_states
		else:
			alpha = np.clip(alpha, 0.1, 0.9)

		rs = tf.linalg.sqrtm(cov).eval()

		pointss = np.einsum('aij,js->ais', rs, np.array([np.cos(self._t), np.sin(self._t)]))
		pointss += loc[:, :, None]

		for i, c, a in zip(range(0, nb_states, nb), color, alpha):
			points = pointss[i]

			if edgecolor is None:
				edgecolor = c

			polygon = plt.Polygon(points.transpose().tolist(), facecolor=c, alpha=a,
								  linewidth=linewidth, zorder=zorder,
								  edgecolor=edgecolor)

			if edgealpha is not None:
				plt.plot(points[0, :], points[1, :], color=edgecolor)

			if ax:
				ax.add_patch(polygon)  # Patch

				l = None
				if center:
					a = alpha[i]
				else:
					a = 0.

				ax.plot(loc[i, 0], loc[i, 1], '.', color=c, alpha=a)  # Mean

				if border:
					ax.plot(points[0, :], points[1, :], color=c, linewidth=linewidth,
							markersize=markersize)  # Contour
				if equal:
					ax.set_aspect('equal')
			else:
				if empty:
					plt.gca().grid('off')
					# ax[-1].set_xlabel('x position [m]')
					plt.gca().set_axis_bgcolor('w')
					plt.axis('off')

				plt.gca().add_patch(polygon)  # Patch
				l = None

				if center:
					a = alpha[i]
				else:
					a = 0.0

				l, = plt.plot(loc[i, 0], loc[i, 1], '.', color=c, alpha=a)  # Mean
				# plt.plot(points[0,:], points[1,:], color=c, linewidth=linewidth , markersize=markersize) # Contour
				if equal:
					plt.gca().set_aspect('equal')
		return l

def plot_coordinate_system(A, b, scale=1., equal=True, ax=None, **kwargs):
	"""

	:param A:		nb_dim x nb_dim
		Rotation matrix
	:param b: 		nb_dim
		Translation
	:param scale: 	float
		Scaling of the axis
	:param equal: 	bool
		Set matplotlib axis to equal
	:param ax: 		plt.axes()
	:param kwargs:
	:return:
	"""
	a0 = np.vstack([b, b + scale * A[:,0]])
	a1 = np.vstack([b, b + scale * A[:,1]])

	if ax is None:
		p, a = (plt, plt.gca())
	else:
		p, a = (ax, ax)

	if equal and a is not None:
		a.set_aspect('equal')

	p.plot(a0[:, 0], a0[:, 1], 'r', **kwargs)
	p.plot(a1[:, 0], a1[:, 1], 'b', **kwargs)

def plot_coordinate_system_3d(
		A, b, scale=1., equal=True, dim=None, ax=None, text=None, dx_text=[0., 0.],
		text_kwargs={}, **kwargs):
	"""

	:param A:		nb_dim x nb_dim
		Rotation matrix
	:param b: 		nb_dim
		Translation
	:param scale: 	float
		Scaling of the axis
	:param equal: 	bool
		Set matplotlib axis to equal
	:param ax: 		plt.axes()
	:param kwargs:
	:return:
	"""

	if dim is None:
		dim = [0, 1]

	a0 = np.vstack([b[dim], b[dim] + scale * A[dim, 0]])
	a1 = np.vstack([b[dim], b[dim] + scale * A[dim, 1]])
	a2 = np.vstack([b[dim], b[dim] + scale * A[dim, 2]])

	if ax is None:
		p, a = (plt, plt.gca())
	else:
		p, a = (ax, ax)


	if equal and a is not None:
		a.set_aspect('equal')


	if text is not None:
		a.text(b[dim[0]] + dx_text[0], b[dim[1]] + dx_text[1], text, **text_kwargs)

	label = kwargs.pop('label', None)
	color = kwargs.get('color', None)
	if label is not None: kwargs['label'] = label + ' x'
	if color is not None: kwargs['label'] = label

	p.plot(a0[:, 0], a0[:, 1], 'r', **kwargs)

	if color is not None:
		label = kwargs.pop('label', None)

	if label is not None and color is None: kwargs['label'] = label + ' y'
	p.plot(a1[:, 0], a1[:, 1], 'g', **kwargs)
	if label is not None and color is None: kwargs['label'] = label + ' z'
	p.plot(a2[:, 0], a2[:, 1], 'b', **kwargs)
