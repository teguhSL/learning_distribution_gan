import numpy as np
import os, sys

def load_letter(letter='P', get_x=True, get_dx=True, get_ddx=False, concat=True,
				fill_zeros=None):
	path = os.path.abspath(__file__)

	datapath = path[:-38]
	if datapath[-1] != '/':
		datapath += '/'

	datapath += 'data/2Dletters/'

	try:
		data = np.load(datapath + '%s.npy' % letter, allow_pickle=True)[()]
	except ValueError:
		data = np.load(datapath + '%s_2.npy' % letter, allow_pickle=True)[()]
	except:
		print("Unexpected error:", sys.exc_info()[0])
		raise

	demos = []

	if get_x:
		if fill_zeros is None:
			demos += [data['x']]
		else:
			demos += [[np.concatenate(
				[d, d[[-1]] * np.ones((fill_zeros, data['x'][0].shape[-1]))], axis=0)
				for d in data['x']]]

	if get_dx:
		if fill_zeros is None:
			demos += [data['dx']]
		else:
			demos += [[np.concatenate(
				[d, np.zeros((fill_zeros, data['dx'][0].shape[-1]))], axis=0)
			for d in data['dx']]]

	if get_ddx:
		if fill_zeros is None:
			demos += [data['ddx']]
		else:
			demos += [[np.concatenate(
				[d, np.zeros((fill_zeros, data['ddx'][0].shape[-1]))], axis=0)
			for d in data['ddx']]]

	if concat:
		return np.concatenate(demos, axis=2)
	else:
		return demos

def load_letter_bimodal(letter='P', get_x=True, get_dx=True, get_ddx=False, concat=True,
				fill_zeros=None, center=[0.2, 0.], vec=[1, 0.], dist=0.1, displ=0.2):

	demos = load_letter(letter=letter, get_x=get_x, get_dx=get_dx, get_ddx=get_ddx,
						concat=False, fill_zeros=fill_zeros)

	direction = 2. * (np.random.randint(0, 2, len(demos[0])) - 0.5)

	act = np.exp(-np.sum(((np.array(demos[0]) - center) / dist) ** 2, axis=-1))

	d_x = direction[:, None, None] * displ * np.array(vec)[None, None, :] * act[:, :, None]
	demos[0] = np.array(demos[0]) + d_x

	if get_dx:
		demos[1] = np.array(demos[1]) + np.gradient(d_x, axis=1) / 0.01
	if get_ddx:
		demos[2] = np.array(demos[2]) + np.gradient(np.gradient(d_x, axis=1), axis=1) / 0.01 ** 2

	if concat:
		return np.concatenate(demos, axis=2)
	else:
		return demos