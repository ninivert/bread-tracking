from ._utils import SplitDataModule, AssignmentGraphDataset, InplaceAttrRecastTransform

import torchvision.transforms as transforms
import torch
import pytorch_lightning as pl
import os

__all__ = ['train']

LIGHTNING_DIR = 'model_logs'

def train(model_cls, config, data_filepaths, data_transforms=None):
	"""Train a model, initialized with a given config

	Parameters
	----------
	model_cls : NodeClassifierBase
		Node classifier model
	config : dict
		configuration for the model and training parameters
	data_filepaths : List[str]
		location of the data .npz files
	data_transforms : None, optional
		optional runtime transforms to apply to the data

	Returns
	-------
	model
		optimized model
	"""

	pl.seed_everything(config['seed'], workers=True)

	model = model_cls(config)

	if data_transforms is None:
		data_transforms = transforms.Compose([
			InplaceAttrRecastTransform('y', torch.long),
			# EqualizeTransform()
		])

	datamodule = SplitDataModule(
		AssignmentGraphDataset(
			data_filepaths,
			preload=True,
			transform=data_transforms
		),
		dataloader_kwargs=dict(num_workers=12, batch_size=config['batch_size']),
		mask_fractions=config['dataset_split']
	)

	print(model)

	trainer = pl.Trainer(
		default_root_dir=LIGHTNING_DIR,
		max_epochs=config['num_epochs'],
		gpus=1,
		enable_model_summary=True,
		callbacks=[
			pl.callbacks.progress.RichProgressBar(),
			pl.callbacks.ModelCheckpoint(monitor='ptl/val_loss'),
		],
		logger=pl.loggers.TensorBoardLogger(
			save_dir=os.path.join(LIGHTNING_DIR, model_cls.__name__.replace('NodeClassifier', '')),
			name=' '.join(f'{key}={value}' for key, value in config.items()),
			default_hp_metric=False,
		),
	)

	trainer.fit(model, datamodule)

	return model, trainer, datamodule
