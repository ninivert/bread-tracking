import matplotlib.pyplot as plt
import matplotlib.lines
from bread.utils import get_cms, get_cellids
import numpy as np

__all__ = ['plot_visible', 'plot_seg', 'plot_graph', 'plot_ellipsefits']

def plot_visible(img_vis, figax=None, **kwargs):
	if figax is None:
		fig, ax = plt.subplots(figsize=(4, 4))
	else:
		fig, ax = figax
		
	ax.imshow(img_vis, cmap='binary', **kwargs)
	
	return fig, ax


def plot_seg(img_seg, figax=None, **kwargs):
	if figax is None:
		fig, ax = plt.subplots(figsize=(4, 4))
	else:
		fig, ax = figax
	
	img_seg_ = img_seg.astype(float)
	img_seg_[img_seg_ == 0] = np.nan
	ax.imshow(img_seg_, cmap='gist_rainbow', **kwargs)
	
	cell_ids = get_cellids(img_seg)
	cms = get_cms(img_seg)
	for cm, cell_id in zip(cms, cell_ids):
		ax.text(cm[1], cm[0], f'{int(cell_id)}', fontsize=8, ha='center', va='center')
		
	fig.tight_layout()
	
	return fig, ax


def plot_graph(img_seg, edges, figax=None):
	if figax is None:
		fig, ax = plt.subplots(figsize=(4, 4))
	else:
		fig, ax = figax
		
	cell_ids = get_cellids(img_seg)
	cms = get_cms(img_seg)
	
	for edge in edges:
		cm1 = cms[np.where(cell_ids == edge[0])[0]][0]
		cm2 = cms[np.where(cell_ids == edge[1])[0]][0]
		ax.add_artist(matplotlib.lines.Line2D(
			(cm1[1], cm2[1]), (cm1[0], cm2[0]),
			color='black', linewidth=1, alpha=0.7
		))
		
	return fig, ax


def plot_ellipsefits(img_seg, r_majs, r_mins, alphas, figax=None, draw_axes=True):
	if figax is None:
		fig, ax = plt.subplots(figsize=(4, 4))
	else:
		fig, ax = figax
		
	cell_ids = get_cellids(img_seg)
	cms = get_cms(img_seg)
	
	for cm, r_maj, r_min, alpha in zip(cms, r_majs, r_mins, alphas):
		rot = np.array(((np.sin(alpha), -np.cos(alpha)), (np.cos(alpha), np.sin(alpha))))
		vec_maj = cm + rot @ np.array((r_maj, 0))
		vec_min = cm + rot @ np.array((0, r_min))
		if draw_axes:
			ax.add_artist(matplotlib.lines.Line2D(
				(cm[1], vec_maj[1]), (cm[0], vec_maj[0]),
				color='black', linewidth=1, alpha=0.8
			))
			ax.add_artist(matplotlib.lines.Line2D(
				(cm[1], vec_min[1]), (cm[0], vec_min[0]),
				color='black', linewidth=1, alpha=0.8
			))
		ax.add_artist(matplotlib.patches.Ellipse(
			(cm[1], cm[0]), 2*r_maj, 2*r_min, np.rad2deg(alpha),
			color='black', alpha=0.8, fill=False
		))
	
	return fig, ax
