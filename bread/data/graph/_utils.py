import torch
from torch_geometric.data import Data

__all__ = ['from_npz']

def from_npz(npdata) -> Data:
	"""Data is stored as numpy.ndarray in the .npz files. These have to be cast back to torch tensors before being able to be used by pytorch_geometric using this method"""

	if isinstance(npdata, list):
		return [from_npz(x) for x in npdata]

	return Data(**dict((key, torch.from_numpy(item)) for key, item in npdata.items()))
