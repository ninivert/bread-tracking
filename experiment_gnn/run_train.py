from glob import glob

DATA_FILEPATHS = {
	# 'assgraphs_small': sorted(glob('/home/niels/TP4/experiment_gnn/data/assgraphs/*/0*.npz')),
	# 'assgraphs_lesssmall': sorted(glob('/home/niels/TP4/experiment_gnn/data/assgraphs/*/[01][01]*.npz')),
	# 'assgraphs': sorted(glob('/home/niels/TP4/experiment_gnn/data/assgraphs/*/*.npz'))
	'assgraphs_colony0123_framediffall': sorted(glob('../data/assgraphs/colony00[0123]/framediff*/*.npz'))
}

from bread_tracking.algo.tracking.gnn import train, GNNNodeClassifier1, MLPNodeClassifier, AssignmentGraphDataset, InplaceAttrRecastTransform, EqualizeTransform, SplitDataModule
import torch
import torchvision.transforms as transforms
import itertools

encoder_num_layers = 3
hidden_channels = 80
num_layers = 8

config = dict(
	# constant parameters
	node_dim=6,  # 6 features per node
	num_classes=2,  # binary classification
	edge_dim=6,  # 6 features per edge
	# model hyperparameters
	encoder_num_layers=encoder_num_layers,
	hidden_channels=hidden_channels,
	num_layers=num_layers,
	dropout=0.1,
	# training parameters
	lr=1e-4,
	lr_decay=10**(-1/100),  # decay in total 10**(-1) in total after 100 epochs
	seed=0,
	batch_size=32,
	num_epochs=90,
	# dataset parameters
	dataset_name='assgraphs_colony0123_framediffall',
	dataset_split=[0.8, 0.2, 0]
)

train(
	GNNNodeClassifier1,
	config,
	DATA_FILEPATHS[config['dataset_name']]
)

#
# MLP training (hyperparam scan)
#

# for num_layers, hidden_channels in itertools.product((1, 2, 3, 4), (8, 16, 32, 48)):
# 	config = dict(
# 		# constant parameters
# 		node_dim=6,  # 6 features per node
# 		num_classes=2,  # binary classification
# 		# edge_dim=6,  # 6 features per edge -> MLP does not take edge into account
# 		# model hyperparameters
# 		hidden_channels=hidden_channels,
# 		num_layers=num_layers,
# 		dropout=0.1,
# 		# training parameters
# 		seed=0,
# 		lr=1e-4,
# 		lr_decay=10**(-1/100),
# 		batch_size=256,
# 		num_epochs=100,
# 		# dataset parameters
# 		dataset_name='assgraphs_colony0123_framediffall',
# 		dataset_split=[0.8, 0.2, 0]
# 	)

# 	train(
# 		MLPNodeClassifier,
# 		config,
# 		DATA_FILEPATHS[config['dataset_name']],
# 		data_transforms=transforms.Compose([
# 			InplaceAttrRecastTransform('y', torch.long),
# 			EqualizeTransform()
# 		])
# 	)

#
# GNN training (hyperparameter scan)
#

# for encoder_num_layers, num_layers, hidden_channels in itertools.product((3, 4, 5), (8, 10, 12, 14), (60, 80, 100, 120)):
# for encoder_num_layers, num_layers, hidden_channels in itertools.product((3, 2, 1), (4, 3, 2, 1), (60, 30)):
# 	print('--------------------------------------------------------------------------------')
# 	print(f'{encoder_num_layers=}, {num_layers=}, {hidden_channels=}')
# 	if (encoder_num_layers, num_layers, hidden_channels) in [(5,11,120), (5,11,90), (5,11,60), (5,8,120), (5,8,90), (5,8,60), (5,5,120), (5,5,90), (5,5,60), (5,2,120), (5,2,90), (5,2,60), (4,8,120), (4,8,90)]:
# 		print('skipping since already done')
# 		continue

# 	config = dict(
# 		# constant parameters
# 		node_dim=6,  # 6 features per node
# 		num_classes=2,  # binary classification
# 		edge_dim=6,  # 6 features per edge
# 		# model hyperparameters
# 		encoder_num_layers=encoder_num_layers,
# 		hidden_channels=hidden_channels,
# 		num_layers=num_layers,
# 		dropout=0.1,
# 		# training parameters
# 		lr=1e-4,
# 		lr_decay=10**(-1/100),  # decay in total 10**(-1) in total after 100 epochs
# 		seed=0,
# 		batch_size=32,
# 		num_epochs=90,
# 		# dataset parameters
# 		dataset_name='assgraphs_colony0123_framediffall',
# 		dataset_split=[0.8, 0.2, 0]
# 	)

# 	train(
# 		GNNNodeClassifier1,
# 		config,
# 		DATA_FILEPATHS[config['dataset_name']]
# 	)
