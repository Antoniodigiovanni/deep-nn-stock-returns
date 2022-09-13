from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance
from data.dataset import BaseDataset
from data.crsp_dataset import CrspDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch

def IntegratedGradients_importance(model, path):
    ig = IntegratedGradients(model)
    dataset = BaseDataset()
    df = dataset.df

    feature_names = list(df.drop(['permno','ret','yyyymm','melag','me','prc','me_nyse20'], axis=1).columns)
    print(feature_names)
    df = df.loc[df['yyyymm'] >= 199501]
    df = CrspDataset(df)
    test_loader = DataLoader(df, batch_size=100)

    dataiter = iter(test_loader)
    inputs, target, labels = dataiter.next()
    print(f'Inputs has type: {type(inputs)} and shape {inputs.size()}')
    test_input_tensor = inputs.float()
    test_input_tensor.requires_grad_()
    attr, delta = ig.attribute(test_input_tensor, return_convergence_delta=True)
    attr = attr.detach().numpy()

    visualize_importances(feature_names, np.mean(attr, axis=0), path)

def visualize_importances(feature_names, importances, path, title="Average Feature Importances", plot=True, axis_title="Features"):
    print(title)
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", '%.3f'%(importances[i]*1000000000))
    x_pos = (np.arange(len(feature_names)))
    if plot:
        # plt.figure(figsize=(12,6))
        # plt.bar(x_pos, importances, align='center')
        # plt.xticks(x_pos, feature_names, wrap=True)
        # plt.xlabel(axis_title)
        # plt.title(title)
        # plt.savefig("feature_importance.png")

        plt.figure(figsize=(50,20))
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names, wrap=True)
        plt.xlabel(axis_title)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(path)

