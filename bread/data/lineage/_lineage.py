import pandas as pd
import numpy as np
from typing import Union
from collections import namedtuple

__all__ = ['Lineage', 'load_lineage', 'load_lineage_pd']

Lineage = namedtuple('Lineage', ('parent_ids', 'bud_ids', 'time_ids'))

def load_lineage(filepath: Union[str, list[str]]) -> Lineage:
	if isinstance(filepath, list):
		return [load_lineage(x) for x in filepath]
	
	parent_ids, bud_ids, time_ids = np.genfromtxt(filepath, skip_header=True, delimiter=',', unpack=True)

	return Lineage(parent_ids, bud_ids, time_ids)


def load_lineage_pd(filepath: Union[str, list[str]]):
	if isinstance(filepath, list):
		return [load_lineage(x) for x in filepath]
	
	return pd.read_csv(filepath)
