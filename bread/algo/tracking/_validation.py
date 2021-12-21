__all__ = ['accuracy']

def accuracy(A1, A2):
	"""Compute similarity score between two matrices

	Parameters
	----------
	A1 : array-like of shape (n1, n2)
		Predicted assignment matrix
	A2 : array-like of shape (n1, n2)
		Ground truth assignment matrix

	Returns
	-------
	accuracy: float
		accuracy score, in range [0, 1]
	"""

	return (A1 == A2).sum() / (A1.shape[0] * A1.shape[1])
