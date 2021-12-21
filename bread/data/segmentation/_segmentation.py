import numpy as np
import h5py
from typing import Union

__all__ = ['load_segmentations_yeaz']

def load_segmentations_yeaz(filepath: Union[str, list[str]], fov='FOV0'):
	if isinstance(filepath, list):
		return [load_segmentations_yeaz(x, fov) for x in filepath]
	
	file = h5py.File(filepath, 'r')
	imgs = np.zeros((len(file[fov].keys()), *file[fov]['T0'].shape), dtype=int)
	for i in range(len(file['FOV0'])):
		imgs[i] = np.array(file['FOV0'][f'T{i}'])
	file.close()
	
	return imgs
