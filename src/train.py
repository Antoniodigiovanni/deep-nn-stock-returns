from trainer.expanding_window_trainer import ExpandingWindowTraining
from data.base_dataset import BaseDataset
from models.neural_net.regression_net import RegressionNet
data = BaseDataset()
data.load_dataset_in_memory()
dataset = data.crsp


net = ExpandingWindowTraining(dataset)
net.fit()
