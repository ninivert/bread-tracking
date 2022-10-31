"""Load an old checkpoint and convert the state_dict to pytorch lightning new state dict"""

from copy import deepcopy
import torch
import sys
sys.path.append('../..')
from bread_tracking.algo.tracking.gnn import GNNNodeClassifier1

old_checkpoint = torch.load('GNN1_demo.lightning==1.5.5.ckpt', map_location=torch.device('cpu'))
old_state_dict = old_checkpoint['state_dict']

# config = {'node_dim': 6, 'num_classes': 2, 'edge_dim': 6, 'encoder_num_layers': 5, 'hidden_channels': 120, 'num_layers': 11, 'dropout': 0.1, 'lr': 0.0001, 'lr_decay': 0.9772372209558107, 'seed': 0, 'batch_size': 20, 'num_epochs': 90, 'dataset_name': 'assgraphs_colony0123_framediffall', 'dataset_split': [0.8, 0.2, 0]}
config = old_checkpoint['hyper_parameters']

model = GNNNodeClassifier1(config)

new_state_dict = model.state_dict()

for knew, kold in zip(new_state_dict, old_state_dict):
	print('{:<60} {:<60}'.format(knew, kold))
	new_state_dict[knew] = old_state_dict[kold]

model.load_state_dict(new_state_dict)
new_state_dict = model.state_dict()

new_checkpoint = deepcopy(old_checkpoint)
new_checkpoint['pytorch-lightning_version'] = '1.7.5'
new_checkpoint['state_dict'] = new_state_dict

torch.save(new_checkpoint, 'GNN1_demo.lightning==1.7.5.pt')
