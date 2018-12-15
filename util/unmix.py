import numpy as np
from cvxopt import matrix
from cvxopt import solvers

def unmix(data, endmembers):
	X = data

	n_endmember = endmembers.shape[1]
	n_pixel = X.shape[1]

	# equality constraint A * x = b
	# all values must sum to 1 (X1 + X2 + ... + XM = 1)
	A = matrix(np.ones((1,n_endmember)), tc='d')
	b = matrix(1.0, tc='d')

	# boundary constraints lb >= x >= ub
	# All values must be greater than 0 (0 < X1, 0 < X2 ... 0 < XM)
	G = np.zeros((2 * n_endmember, n_endmember))
	G[:n_endmember,:] = -np.eye(n_endmember)
	G[n_endmember:,:] = np.eye(n_endmember)
	h = np.zeros(2 * n_endmember)
	h[n_endmember:] = 1.0

	G = matrix(G, tc='d')
	h = matrix(h, tc='d')

	H = matrix(np.float64(2 * (endmembers.T @ endmembers)))
	P = np.zeros((n_pixel,n_endmember))

	solvers.options['show_progress'] = False
	for i in range(n_pixel):
		F = matrix(np.float64((-2 * X[:,i][np.newaxis,:] @ endmembers).T), tc='d')
		qp_out = solvers.qp(H, F, G, h, A, b)
		P[i,:] = np.array(qp_out['x']).T
		# print(np.array(qp_out['x']).shape)
	P[P<0] = 0

	return P
