import networkx as nx
import torch_geometric
from torch_geometric.data import Data
import itertools
import numpy as np
import matplotlib.pyplot as plt
import bread_tracking.data.graph
import bread_tracking.utils

__all__ = ['plot_graph_networkx', 'plot_assignmentgraph_track', 'plot_assignmentgraph']

def plot_graph_networkx(graph: Data, undirected=True, **kwargs):
	if undirected:
		edge_index_undir = torch_geometric.utils.to_undirected(graph.edge_index)
		graph_ = Data(x=graph.x, edge_index=edge_index_undir)
	else:
		graph_ = graph
	
	nx.draw(
		torch_geometric.utils.to_networkx(graph_, to_undirected=undirected),
		**kwargs
	)
	
def plot_assignmentgraph_track(img_seg1, img_seg2, g1, g2):
	gA = bread_tracking.data.graph.build_assignmentgraph(g1, g2)
	gAt = bread_tracking.data.graph.build_assignmentgraph_track(img_seg1, img_seg2)
	
	plt.close('all')
	fig, ax = plot_assignmentgraph(
		g1, g2, gA,
		kwargs_g1=dict(pos=bread_tracking.utils.get_cms(img_seg1)[:, [1, 0]]),
		kwargs_g2=dict(pos=bread_tracking.utils.get_cms(img_seg2)[:, [1, 0]]),
		kwargs_gA=dict(node_color=['#0f0' if tracklink else '#f00' for tracklink in gAt.x])
	)
	ax[0].invert_yaxis()
	ax[1].invert_yaxis()
	
	return fig, ax
	
	

def plot_assignmentgraph(g1, g2, gA, kwargs_g1={}, kwargs_g2={}, kwargs_gA={}):
	labels1 = dict((i, f'{i+1:d}') for i in range(g1.num_nodes))
	labels2 = dict((a, f'{chr(ord("a")+a)}') for a in range(g2.num_nodes))
	labelsA = dict(
		(
			np.ravel_multi_index(((i,), (a,)), (g1.num_nodes, g2.num_nodes))[0],
			labels1[i] + labels2[a]
		)
		for i, a in itertools.product(range(g1.num_nodes), range(g2.num_nodes))
	)
	
	fig, ax = plt.subplots(figsize=(12, 4), ncols=3, gridspec_kw={'width_ratios': [2, 2, 4]})
	ax[0].set_title('$G^{(1)}$')
	ax[1].set_title('$G^{(2)}$')
	ax[2].set_title('$G^{(A)}$')
	plt.sca(ax[0])
	bread_tracking.vis.graph.plot_graph_networkx(g1, labels=labels1, **kwargs_g1)
	plt.sca(ax[1])
	bread_tracking.vis.graph.plot_graph_networkx(g2, labels=labels2, **kwargs_g2)
	plt.sca(ax[2])
	bread_tracking.vis.graph.plot_graph_networkx(gA, labels=labelsA, **kwargs_gA)
	fig.tight_layout()
	
	return fig, ax
