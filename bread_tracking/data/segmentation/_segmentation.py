from pathlib import Path
import numpy as np
import h5py
from typing import Union, List

__all__ = ['load_segmentations_yeaz', 'save_segmentations_yeaz']

def load_segmentations_yeaz(filepath: Union[str, List[str]], fov='FOV0'):
	if isinstance(filepath, list):
		return [load_segmentations_yeaz(x, fov) for x in filepath]
	
	file = h5py.File(filepath, 'r')
	imgs = np.zeros((len(file[fov].keys()), *file[fov]['T0'].shape), dtype=int)
	for i in range(len(file[fov])):
		imgs[i] = np.array(file[fov][f'T{i}'])
	file.close()
	
	return imgs

def save_segmentations_yeaz(seg, filepath: Path, fov='FOV0'):
	file = h5py.File(filepath, 'w')
	file.create_group(fov)
	for i in range(len(seg)):
		file.create_dataset(f'{fov}/T{i}', data=seg[i], compression='gzip')
	file.close()