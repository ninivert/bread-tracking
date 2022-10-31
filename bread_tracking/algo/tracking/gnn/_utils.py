import bread_tracking.data.graph
import bread_tracking.data

import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

import numpy as np

__all__ = ['EqualizeTransform', 'InplaceAttrRecastTransform', 'AssignmentGraphDataset', 'SplitDataModule']

class EqualizeTransform:
	"""Implements equalization of class labels in a batch"""

	def __call__(self, obj):
		x, y = obj.x, obj.y

		y_uniques, y_counts = np.unique(y, return_counts=True)
		y_count_thresholder = np.min(y_counts)

		mask_points = np.ones(x.shape[0], dtype=bool)
		for y_unique, y_count in zip(y_uniques, y_counts):
			nb_rm = abs(y_count - y_count_thresholder)  # number of points to remove
			mmask = np.ones(y_count, dtype=bool)
			mmask[:nb_rm] = False
			mask_points[y == y_unique] = np.random.permutation(mmask)

		return Data(x=x[mask_points, :], y=y[mask_points])


class InplaceAttrRecastTransform:
	"""Recast a given attribute to a given type"""

	def __init__(self, attr: str, dtype=torch.long):
		self.dtype = dtype
		self.attr = attr

	def __call__(self, obj):
		x = getattr(obj, self.attr)
		x = x.type(self.dtype)
		setattr(obj, self.attr, x)
		return obj


class AssignmentGraphDataset(Dataset):
	"""Loads graphs from .npz files into the Data format from torch_geometric"""

	def __init__(self, filepaths, preload=False, transform=None):
		super().__init__()
		self.filepaths = filepaths

		if preload:
			self.graph_cache = [bread_tracking.data.graph.from_npz(bread_tracking.data.utils.load_npz(self.filepaths[idx])) for idx in range(len(self))]
		self.preload = preload

		self.transform = None  # set to None because we need to initialize the following with __getitem__\
		self.num_node_features = self[0].num_node_features
		self.num_edge_features = self[0].num_edge_features
		self.num_classes = 2  # Binary classification
		self.transform = transform

	def __len__(self):
		return len(self.filepaths)

	def __getitem__(self, idx):
		if self.preload:
			graph = self.graph_cache[idx]
		else:
			graph = bread_tracking.data.graph.from_npz(bread_tracking.data.utils.load_npz(self.filepaths[idx]))

		if self.transform is not None:
			if isinstance(graph, list):  # handle index slices
				for i in range(len(graph)):
					graph[i] = self.transform(graph[i])
			else:
				graph = self.transform(graph)

		return graph


class SplitDataModule(pl.LightningDataModule):
	"""Split a dataset into dataloaders according to mask_fractions"""

	def __init__(self, dataset: Dataset, mask_fractions=[0.9, 0.1], dataloader_kwargs={'batch_size': 1}, **kwargs):
		super().__init__(self, **kwargs)
		self.dataset = dataset
		self.dataloader_kwargs = dataloader_kwargs
		self.batch_size = dataloader_kwargs['batch_size']

		randidx = torch.randperm(len(self.dataset))
		mask_idx = [int(sum(mask_fractions[:i+1])*len(self.dataset)) for i in range(len(mask_fractions))]
		self.train_mask_idx = randidx[:mask_idx[0]]
		self.val_mask_idx = randidx[mask_idx[0]:mask_idx[1]]
		self.test_mask_idx = randidx[mask_idx[1]:]

	def train_dataloader(self):
		return DataLoader(Subset(self.dataset, self.train_mask_idx), **self.dataloader_kwargs)

	def val_dataloader(self):
		return DataLoader(Subset(self.dataset, self.val_mask_idx), **self.dataloader_kwargs)

	def test_dataloader(self):
		return DataLoader(Subset(self.dataset, self.test_mask_idx), **self.dataloader_kwargs)
