import sys, os

# To import config from top_level folder
currentPath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentPath+'/../')

from data.base_dataset import BaseDataset
from models.neural_net.metric import accuracy
#from config import config
from nni.retiarii.oneshot.pytorch import DartsTrainer

data = BaseDataset()
data.load_split_train_data()

trainer = DartsTrainer(
   model=model,
   loss=criterion,
   metrics=lambda output, target: accuracy(target, output, 0.1),
   optimizer=optim,
   dataset=dataset_train,
   batch_size=32,
   log_frequency=50
)
trainer.fit()