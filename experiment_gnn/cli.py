if __name__ == '__main__':
	import logging
	logging.basicConfig()
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)

	import argparse
	from pathlib import Path

	parser = argparse.ArgumentParser()

	parser.add_argument('input', type=Path, help='path to input segmentation file')
	parser.add_argument('output', type=Path, help='path to output segmentation file')
	parser.add_argument('--from', required=True, dest='ifrom', type=int, help='start frame for tracking')
	parser.add_argument('--to', required=True, dest='ito', type=int, help='end frame for tracking')
	# TODO : exclusive option to retrack all

	args = parser.parse_args()

	logging.info(args)

	import sys
	sys.path.append('../')
	from bread_tracking.data.segmentation import load_segmentations_yeaz, save_segmentations_yeaz
	from bread_tracking.data.graph import build_cellgraph, build_assignmentgraph
	from bread_tracking.algo.tracking.gnn import GNNNodeClassifier1, assmatrix_linear_sum, assmatrix_to_track
	from bread_tracking.utils import get_cellids
	import torch

	logger.info(f'Loading segmentation from `{args.input}`...')
	seg = load_segmentations_yeaz(args.input)
	
	assert 0 <= args.ifrom and args.ifrom < len(seg), 'invalid starting frame index'
	assert 0 <= args.ito and args.ito < len(seg), 'invalid ending frame index'

	logger.info('Building cellgraph...')
	graph1 = build_cellgraph(seg[args.ifrom], max_membdist=12)
	graph2 = build_cellgraph(seg[args.ito], max_membdist=12)

	logger.info('Building assignment graph...')
	assgraph = build_assignmentgraph(graph1, graph2)

	logger.info('Running the GNN...')
	model = GNNNodeClassifier1.load_from_checkpoint('model_weights/best/best.lightning==1.7.5.pt')
	model.eval()
	with torch.no_grad():
		z = model(assgraph).detach().reshape(graph1.num_nodes, graph2.num_nodes, 2)

	logger.info('Optimizing assignments...')
	ass = assmatrix_linear_sum(z)
	track = assmatrix_to_track(ass)

	seg_track = seg[args.ito].copy()
	cell_ids1 = get_cellids(seg[args.ifrom])

	def find_new_cell_id(seg_prior, seg_current) -> int:
		"""Return a cell id that hasn't been assigned yet
		
		seg_prior :
			previous frames
		seg_current :
			the segmentation being currently modified
		"""
		return max(seg_prior.max(), seg_current.max()) + 1

	for idx2, cell_id2 in enumerate(get_cellids(seg[args.ito])):
		y, x = (seg[args.ito] == cell_id2).nonzero()  # indices of the cell we want to backtrack

		# predicted track
		idx1 = int(track[idx2])
		# -1 encodes a new cell
		new_cell_id2 = cell_ids1[idx1] if idx1 >= 0 else -1
		if new_cell_id2 == -1:
			print('# new cell')
			new_cell_id2 = find_new_cell_id(seg[:args.ito], seg_track)
		seg_track[y, x] = new_cell_id2

		print('{} -> {}'.format(cell_id2, new_cell_id2))

	seg[args.ito] = seg_track

	logger.info(f'Saving segmentation to `{args.output}`...')
	save_segmentations_yeaz(seg, args.output)

	logger.info('Done')