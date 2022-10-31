import numpy as np
import pandas as pd
import itertools
import cv2 as cv
import scipy.spatial
from bread_tracking.utils import get_cellids, get_cms, cell_contour

__all__ = ['extract_features_cells', 'extract_features_pairs']

def extract_features_cells(img_seg, cell_ids=None):
	"""Compute geometric features of individual cells from a segmentation

	Parameters
	----------
	img_seg : array-like of shape (W, H)
		segmentation
	cell_ids : None, optional
		list of cell ids for which to compute the features. Defaults to all cells in the segmentation

	Returns
	-------
	df_x
		pandas.DataFrame containing the node features
	"""

	if cell_ids is None:
		cell_ids = get_cellids(img_seg)

	# Precompute the cell contours

	contours = []
	for cell_id in cell_ids:
		contours.append(cell_contour(img_seg == cell_id))

	# Extract node features

	As, r_equivs, r_majs, r_mins, alpha_majs, es = [], [], [], [], [], []
	for i, (cell_id, contour) in enumerate(zip(cell_ids, contours)):
		As.append(cv.contourArea(contour))
		# approximate the cell as an ellipse
		xy, wh, angle_min = cv.fitEllipse(contour)
		r_min, r_maj = wh[0]/2, wh[1]/2
		assert r_min <= r_maj
		r_mins.append(r_min)
		r_majs.append(r_maj)
		r_equivs.append(np.sqrt(r_maj*r_min))  # radius of the circle with same area as ellipse : r=sqrt(a*b)
		alpha_majs.append(np.mod(np.deg2rad(angle_min) + np.pi/2, np.pi))  # angle of the major axis
		es.append(np.sqrt(1 - (r_min/r_maj)**2))  # eccentricity

	df_x = pd.DataFrame({'cell_id': cell_ids, 'A': As, 'r_equiv': r_equivs, 'r_maj': r_majs, 'r_min': r_mins, 'alpha_maj': alpha_majs, 'e': es})

	return df_x


def extract_features_pairs(img_seg, pairs=None):
	"""Compute geometric features of cell-cell relations from a segmentation

	Parameters
	----------
	img_seg : array-like of shape (W, H)
		segmentation
	pairs : None, optional
		list of pairs for which to compute edge features. Defaults to all possible pairs

	Returns
	-------
	df_e
		pandas.DataFrame containing the edge features
	"""

	cell_ids = get_cellids(img_seg)
	cms = get_cms(img_seg)

	# Precompute the cell contours
	# TODO : we can drop the opencv dependency by computing the closest point to the mask
	# furthermore, this prevents the issue with weird contours (masks with one pixel, etc)

	contours = []
	for cell_id in cell_ids:
		contours.append(cell_contour(img_seg == cell_id))

	# Precompute membrane distances

	memb_dists = np.zeros((len(cell_ids), len(cell_ids)))
	for (i, cell_id1), (j, cell_id2) in itertools.product(enumerate(cell_ids), enumerate(cell_ids)):
		if j <= i: continue
		memb_dists[i, j] = scipy.spatial.distance.cdist(contours[i], contours[j]).min()

	# Extract edge features

	cell_ids1, cell_ids2, rhos, thetas, ls, diff_alphas = [], [], [], [], [], []

	if pairs is None:
		pairs = itertools.combinations(enumerate(cell_ids), 2)

	for (i, cell_id1), (j, cell_id2) in pairs:
		cell_ids1.append(cell_id1)
		cell_ids2.append(cell_id2)
		rhos.append(np.sqrt((cms[i][0] - cms[j][0])**2 + (cms[i][1] - cms[j][1])**2))
		# using mod destroys directional information
		# the nn learns that the angle corresponds to cell1 -> cell2, and not the other way around
		# thetas.append(np.mod(np.arctan2(cms[i][1] - cms[j][1], cms[i][0] - cms[j][0]), np.pi))
		thetas.append(np.arctan2(cms[i][1] - cms[j][1], cms[i][0] - cms[j][0]))
		ls.append(memb_dists[i, j])

		_, _, angle_min1 = cv.fitEllipse(contours[i])
		_, _, angle_min2 = cv.fitEllipse(contours[j])
		angle_min1 = np.mod(np.deg2rad(angle_min1), np.pi)
		angle_min2 = np.mod(np.deg2rad(angle_min2), np.pi)
		diff_alphas.append(np.abs(angle_min1 - angle_min2))

	df_e = pd.DataFrame({'cell_id1': cell_ids1, 'cell_id2': cell_ids2, 'rho': rhos, 'theta': thetas, 'l': ls, 'diff_alpha': diff_alphas})

	return df_e
