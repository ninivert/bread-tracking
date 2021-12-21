import torch
import torch.nn as nn
import torch_geometric as torchg
import pytorch_lightning as pl

__all__ = ['NodeClassifierBase', 'MLPNodeClassifier', 'GNNNodeClassifier1']


class NodeClassifierBase(pl.LightningModule):
	"""Base class for doing node classification

	Attributes
	----------
	accuracy : Callable
		computes accuracy of prediction
	loss : Callable
		loss function used for training
	lr : float
		learning rate of the Adam optimizer
	lr_decay : flat
		exponential learning rate decay parameter
	"""

	def __init__(self, config):
		super().__init__()
		self.save_hyperparameters(config)

		self.lr = config['lr']
		self.lr_decay = config['lr_decay']
		self.loss = nn.CrossEntropyLoss()
		self.accuracy = lambda yhat, y: (y == yhat).sum().float() / y.shape[0]

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
		lr_scheduler = {
			'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay),
		}
		return [optimizer], [lr_scheduler]

	def on_train_start(self):
		self.logger.log_hyperparams(self.hparams, {"hp/val_acc": 0})

	def training_step(self, train_graph, batch_idx):
		l = self.compute_loss(train_graph)
		self.log('ptl/train_loss', l)
		return l

	def validation_step(self, val_graph, batch_idx):
		l = self.compute_loss(val_graph)
		a = self.compute_accuracy(val_graph)
		self.log('ptl/val_loss', l)
		self.log('ptl/val_err', 1-a)
		self.log('hp/val_acc', a)
		return l

	def predict(self, graph):
		z = self.forward(graph)
		yhat = z.argmax(dim=-1)  # apply sigmoid ? -> useless, since it conserves ordering
		return yhat

	def compute_loss(self, graph):
		z = self.forward(graph)
		return self.loss(z, graph.y)

	def compute_accuracy(self, graph):
		return self.accuracy(self.predict(graph), graph.y)


class MLPNodeClassifier(NodeClassifierBase):
	"""Baseline MLP classifier. Takes node attributes as input and attempts to predict between classes {0, 1}

	Attributes
	----------
	layers : torch.nn.MLP
		MLP layers
	norm : torch.nn.BatchNorm1
		Batch normalization layer
	"""

	def __init__(self, config):
		super().__init__(config)

		self.norm = torch.nn.BatchNorm1d(config['node_dim'], track_running_stats=False)
		self.layers = torchg.nn.MLP(
			channel_list=[config['node_dim']] + [config['hidden_channels']]*config['num_layers'] + [config['num_classes']]
		)

	def forward(self, graph, *args, **kwargs):
		return self.layers(self.norm(graph.x))


class GNNNodeClassifier1(NodeClassifierBase):
	"""Graph neural network node classifier.

	based on : https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_proteins_deepgcn.py

	Attributes
	----------
	edge_encoder :
		edge feature encoder. maps edge_dim -> hidden_channels
	node_encoder :
		node feature encoder. maps node_dim -> hidden_channels
	layers :
		list of DeepGCNLayer. applies graph convolution
	lin :
		final linear layer. maps hidden_channels -> 2
	"""

	def __init__(self, config):
		super().__init__(config)

		# self.node_encoder = nn.Linear(config['node_dim'], config['hidden_channels'])
		# self.edge_encoder = nn.Linear(config['edge_dim'], config['hidden_channels'])
		self.node_encoder = torchg.nn.MLP(
			channel_list=[config['node_dim']] + [config['hidden_channels']]*config['encoder_num_layers'],
			dropout=config['dropout']
		)
		self.edge_encoder = torchg.nn.MLP(
			channel_list=[config['edge_dim']] + [config['hidden_channels']]*config['encoder_num_layers'],
			dropout=config['dropout']
		)

		self.layers = nn.ModuleList()

		for i in range(1, config['num_layers'] + 1):
			conv = torchg.nn.GENConv(
				config['hidden_channels'],
				config['hidden_channels'],
				aggr='softmax',
				t=1.0, learn_t=True, num_layers=2, norm='layer'
			)
			norm = nn.LayerNorm(config['hidden_channels'], elementwise_affine=True)
			act = nn.ReLU(inplace=True)

			layer = torchg.nn.DeepGCNLayer(
				conv, norm, act,
				block='res+', dropout=config['dropout'], ckpt_grad=i % 3)

			self.layers.append(layer)

		self.lin = nn.Linear(config['hidden_channels'], config['num_classes'])

	def forward(self, graph, *args, **kwargs):
		x = self.node_encoder(graph.x)
		edge_attr = self.edge_encoder(graph.edge_attr)
		edge_index = graph.edge_index

		x = self.layers[0].conv(x, edge_index, edge_attr)

		for layer in self.layers[1:]:
			x = layer(x, edge_index, edge_attr)

		x = self.layers[0].act(self.layers[0].norm(x))
		x = nn.functional.dropout(x, p=0.1, training=self.training)  # TODO : formalize

		return self.lin(x)
