import numpy as np
import itertools
import scipy.spatial
import torch
from torch import tensor
import torch_geometric
from torch_geometric.data import Data

import bread_tracking.data.features
import bread_tracking.utils

__all__ = ['build_cellgraph', 'build_assignmentgraph', 'build_assignmentgraph_track']

def build_cellgraph(img_seg, max_membdist=12, undirected=False, return_df=False):
	"""Build a cell graph from a segmentation. The resulting graph contains node_features and edge_features, extracted from geometrical properties of the segmentation

	Parameters
	----------
	img_seg : array-like of shape (W, H)
		segmentation
	max_membdist : int, optional
		if the membranes of two cells are separated by less than max_membdist, then they are connected in the graph. Defaults to 12
	undirected : bool, optional
		return an undirected graph instead of the directed graph (to save on memory). Defaults to False
	return_df : bool, optional
		return the intermediary dataframes containing node and edge features

	Returns
	-------
	torch_geometric.data.Data
		cell graph
	"""

	# Node features

	df_x = bread_tracking.data.features.extract_features_cells(img_seg)

	# Edge features
	# we could simply ask to compute all the pairwise features, but that's silly and scales badly

	cell_ids = bread_tracking.utils.get_cellids(img_seg)  # slice to ignore background segmentation, which has id=0
	idmap = bread_tracking.utils.cellid_to_nodeid(cell_ids)

	contours = []
	for cell_id in cell_ids:
		contours.append(bread_tracking.utils.cell_contour(img_seg == cell_id))

	memb_dists = np.zeros((len(cell_ids), len(cell_ids)))
	for (i, cell_id1), (j, cell_id2) in itertools.product(enumerate(cell_ids), enumerate(cell_ids)):
		if j <= i: continue
		memb_dists[i, j] = scipy.spatial.distance.cdist(contours[i], contours[j]).min()

	pairs = []
	for (i, cell_id1), (j, cell_id2) in itertools.product(enumerate(cell_ids), enumerate(cell_ids)):
		if j <= i: continue  # do not put an edge between a cell and itself, and only one vertex per cell-cell pair
		if memb_dists[i, j] > max_membdist: continue  # cells are too far away

		pairs.append((
			(idmap[cell_id1], cell_id1),
			(idmap[cell_id2], cell_id2),
		))

	df_e = bread_tracking.data.features.extract_features_pairs(img_seg, pairs)

	# Build the graph

	# Node features
	dtype = torch.float
	x = torch.stack([
		tensor(df_x.A.to_numpy(), dtype=dtype),
		tensor(df_x.r_equiv.to_numpy(), dtype=dtype),
		tensor(df_x.e.to_numpy(), dtype=dtype)
	], dim=1)

	# Edge indices
	edge_index = torch.zeros((2, len(df_e.cell_id1)), dtype=torch.long)
	for idx, (cell_id1, cell_id2) in df_e.loc[:, ['cell_id1', 'cell_id2']].iterrows():
		edge_index[0, idx] = idmap[cell_id1]
		edge_index[1, idx] = idmap[cell_id2]

	# Edge features
	edge_attr = torch.stack([
		tensor(df_e.rho.to_numpy(), dtype=dtype),
		tensor(df_e.theta.to_numpy(), dtype=dtype),
		tensor(df_e.l.to_numpy(), dtype=dtype)
	], dim=1)

	if undirected:
		# Transform the graph into an undirected graph by copying edge features for the inversed edge indices
		edge_index, edge_attr = torch_geometric.utils.to_undirected(edge_index, edge_attr)

	graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

	# Return

	if return_df:
		return graph, df_x, df_e
	else:
		return graph


def build_assignmentgraph(g1, g2, undirected=True):
	"""Build an assignment graph gA from graphs g1 and g2.
	Assumes sparse representation of an undirected graph, i.e. edges ij with features $\vec{e_ij}$ also encode $\vec{e_ji}$.

	Parameters
	----------
	g1 :
		cell graph corresponding to the first image
	g2 :
		cell graph corresponding to the second image
	undirected : bool, optional
		return an undirected version of the graph. by default, the graph is directed to save on memory

	Returns
	-------
	torch_geometric.data.Data
		assignment graph
	"""

	# build nodes

	x = torch.empty((g1.num_nodes * g2.num_nodes, g1.num_node_features + g2.num_node_features), dtype=torch.float)

	for ia in range(x.shape[0]):
		i, a = np.unravel_index(ia, (g1.num_nodes, g2.num_nodes))
		x[ia] = torch.cat([g1.x[i], g2.x[a]])

	# build edges

	edge_index = torch.empty((2, 2 * g1.num_edges * g2.num_edges), dtype=torch.long)
	edge_attr = torch.empty((edge_index.shape[1], g1.num_edge_features + g2.num_edge_features), dtype=torch.float)

	for ij, ab in itertools.product(range(g1.num_edges), range(g2.num_edges)):
		# node indexes of the ij vertex in graph 1
		i, j = g1.edge_index[:, ij].flatten()
		i, j = int(i), int(j)

		# node indexes of the ij vertex in graph 2
		a, b = g2.edge_index[:, ab].flatten()
		a, b = int(a), int(b)

		# node indexes in the assignment graph
		ia = np.ravel_multi_index(((i,), (a,)), (g1.num_nodes, g2.num_nodes))[0]
		ib = np.ravel_multi_index(((i,), (b,)), (g1.num_nodes, g2.num_nodes))[0]
		ja = np.ravel_multi_index(((j,), (a,)), (g1.num_nodes, g2.num_nodes))[0]
		jb = np.ravel_multi_index(((j,), (b,)), (g1.num_nodes, g2.num_nodes))[0]

		# edge index of the (ia, jb) connection in the assignment graph
		# edge index of the (ib, ja) connection in the assignment graph
		iajb = 2*np.ravel_multi_index(((ij,), (ab,)), (g1.num_edges, g2.num_edges))
		ibja = iajb + 1

		# create edges in the assignment graph
		edge_index[0, iajb], edge_index[1, iajb] = ia, jb
		edge_index[0, ibja], edge_index[1, ibja] = ib, ja

		# edge features in the assignment graph
		edge_feat = torch.cat([g1.edge_attr[ij], g2.edge_attr[ab]])
		edge_attr[iajb] = edge_feat
		edge_attr[ibja] = edge_feat

	if undirected:
		# Make bidirectional
		edge_index, edge_attr = torch_geometric.utils.to_undirected(edge_index, edge_attr)

	return Data(
		x=x,
		edge_index=edge_index,
		edge_attr=edge_attr
	)


def build_assignmentgraph_track(img_seg1, img_seg2) -> Data:
	"""Build an assignment graph to serve as ground truth, using segmentations corresponding to the first and second image

	Parameters
	----------
	img_seg1 : array-like of shape (W, H)
		segmentation for the first image
	img_seg2 : array-like of shape (W, H)
		segmentation for the second image

	Returns
	-------
	torch_geometric.data.Data
		assignment graph
	"""

	cell_ids1 = bread_tracking.utils.get_cellids(img_seg1)
	cell_ids2 = bread_tracking.utils.get_cellids(img_seg2)

	x = torch.empty((len(cell_ids1) * len(cell_ids2), 1), dtype=torch.bool)

	for ia, (cell_id1, cell_id2) in enumerate(itertools.product(cell_ids1, cell_ids2)):
		# node indexes i, a in the graphs 1, 2 respectively
		i, a = np.unravel_index(ia, (len(cell_ids1), len(cell_ids2)))

		# evaluate if cell i (in segmentation 1) is the same as cell a (in segmentation 2)
		x[ia] = bool(cell_ids1[i] == cell_ids2[a])

	return Data(x=x)
