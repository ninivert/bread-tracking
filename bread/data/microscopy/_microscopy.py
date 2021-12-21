import tifffile
from typing import Union

__all__ = ['load_microscopy_raw']

def load_microscopy_raw(filepath: Union[str, list[str]]):
	if isinstance(filepath, list):
		return [load_microscopy_raw(x) for x in filepath]
	
	return tifffile.imread(filepath)
