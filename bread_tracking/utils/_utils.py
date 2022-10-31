import numpy as np
import scipy.ndimage
import cv2 as cv

__all__ = ['cell_contour', 'get_cellids', 'get_cms', 'cellid_to_nodeid']

def cell_contour(mask):
	"""Returns the cell contours from a cell mask

	Parameters
	----------
	mask : numpy.ndarray (shape=(W, H), dtype=bool)
		mask of the cell

	Returns
	-------
	contour : numpy.ndarray (shape=(N, 2), dtype=int)
		indices of the contour of the cell
	"""

	# TODO : check connectivity !!
	# TODO : check if mask is one pixel ! (contour needs to be larger than 5 points to be able to fit an ellipse)
	contours_cv, *_ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
	contours = np.array(contours_cv)[0, :, 0, :]

	return contours


def get_cellids(img_seg, rm_background=True):
	"""Returns cell ids from a segmentation

	Parameters
	----------
	img_seg : array-like of shape (W, H)
		segmentation
	rm_background : bool, optional
		whether to count id=0 as a real cell (YeaZ reserves that id for the background). Defaults to True

	Returns
	-------
	array-like of int
		cell ids contained in the segmentation
	"""
	allids = np.unique(img_seg.flat)

	if rm_background:
		return allids[allids != 0]  # id=0 corresponds to background
	else:
		return allids


def get_cms(img_seg):
	"""Returns centers of mass of cells in a segmentation

	Parameters
	----------
	img_seg : array-list of shape (W, H)
		segmentation

	Returns
	-------
	array-like of shape (ncells, 2)
		coordinates of the centers of mass of each cell
	"""

	cell_ids = get_cellids(img_seg)
	cms = np.zeros((len(cell_ids), 2))

	for i, cell_id in enumerate(cell_ids):
		cms[i] = scipy.ndimage.center_of_mass(img_seg == cell_id)

	return cms


def cellid_to_nodeid(cellids: np.ndarray) -> dict:
	"""Returns a map of cellid to the corresponding nodeid. Loosely equivalent to doing cellids.index(cellid)"""

	return dict((cellid, nodeid) for nodeid, cellid in enumerate(cellids))
