import torch
import scipy.optimize

__all__ = ['assmatrix_naive', 'assmatrix_forward', 'assmatrix_backward', 'assmatrix_backward_constrained', 'assmatrix_linear_sum']

def assmatrix_naive(z):
	"""Construct assignment naively from the GNN output using the "naive" method

	Parameters
	----------
	z : array-like of shape (n1, n2, 2)
		output of the GNN

	Returns
	-------
	A : array-like of shape (n1, n2)
		assignment matrix
	"""

	n1, n2, _ = z.shape
	ass_matrix_naive = z.argmax(axis=2).type(torch.long).squeeze()

	return ass_matrix_naive


def assmatrix_forward(z):
	"""Construct assignment matrix from the GNN output using the "forward" method

	Parameters
	----------
	z : array-like of shape (n1, n2, 2)
		output of the GNN

	Returns
	-------
	A : array-like of shape (n1, n2)
		assignment matrix
	"""

	n1, n2, _ = z.shape
	# compute difference between "yes" and "no"
	z_diff = z.diff(axis=2)
	# see where the model is the least sure about the yes/no distinction
	z_diff_max = z_diff.argmax(axis=1).type(torch.long)
	# convert to one-hot vector
	ass_matrix_fw = torch.zeros(n1, n2, dtype=torch.long).scatter_(1, z_diff_max, 1)

	return ass_matrix_fw


def assmatrix_backward(z):
	"""Construct assignment matrix from the GNN output using the "backward" method

	Parameters
	----------
	z : array-like of shape (n1, n2, 2)
		output of the GNN

	Returns
	-------
	A : array-like of shape (n1, n2)
		assignment matrix
	"""

	n1, n2, _ = z.shape
	# compute difference between "yes" and "no"
	z_diff = z.diff(axis=2)
	# see where the model is the least sure about the yes/no distinction
	z_diff_max = z_diff.argmax(axis=0).type(torch.long)
	# convert to one-hot vector
	ass_matrix_bw = torch.zeros(n2, n1, dtype=torch.long).scatter_(1, z_diff_max, 1)

	return ass_matrix_bw.transpose_(0, 1)


def assmatrix_backward_constrained(z):
	"""Construct assignment matrix from the GNN output using the "backward constrained" method

	Parameters
	----------
	z : array-like of shape (n1, n2, 2)
		output of the GNN

	Returns
	-------
	A : array-like of shape (n1, n2)
		assignment matrix
	"""

	n1, n2, _ = z.shape
	z_diff = z.diff(axis=2)
	# column mask for newly appeared cells
	mask_new = (z_diff < 0).all(dim=0).squeeze()
	# construct backward assignment matrix
	ass_matrix_bw = assmatrix_backward(z)
	# apply corrections : new buds do not have predecessors
	ass_matrix_bw[:, mask_new] = 0

	return ass_matrix_bw


def assmatrix_linear_sum(z):
	"""Construct assignment matrix from the GNN output using the "linear sum optimizer" method

	Parameters
	----------
	z : array-like of shape (n1, n2, 2)
		output of the GNN

	Returns
	-------
	A : array-like of shape (n1, n2)
		assignment matrix
	"""

	n1, n2, _ = z.shape
	cost = z.diff(dim=2).squeeze()
	yx = scipy.optimize.linear_sum_assignment(cost, maximize=True)
	ass_optim = scipy.sparse.bsr_matrix(([1] * len(yx[0]), yx), shape=(n1, n2), dtype=bool).toarray()

	return torch.from_numpy(ass_optim).type(torch.long)
