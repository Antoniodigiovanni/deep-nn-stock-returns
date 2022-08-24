import torch
import sys
import torch.optim as optim


def map_act_func(af_name):
    if af_name == "ReLU":
        act_func = torch.nn.ReLU()
    elif af_name == "LeakyReLU":
        act_func = torch.nn.LeakyReLU()
    elif af_name == "Sigmoid":
        act_func = torch.nn.Sigmoid()
    elif af_name == "Tanh":
        act_func = torch.nn.Tanh()
    elif af_name == "Softplus":
        act_func = torch.nn.Softplus()
    else:
        sys.exit("Invalid activation function")
    return act_func

def map_optimizer(params, net_params):
    opt_name = params['optimizer']['_name']
    lr = params['learning_rate']

    if opt_name == "SGD":
        opt = optim.SGD(net_params, lr=lr, momentum=momentum)
    elif opt_name == "Adam":
        opt = optim.Adam(net_params, lr=lr)
    elif opt_name == "RMSprop":
        momentum = params['optimizer']['momentum']
        print(f'In map optimizer, momentum is: {momentum}')
        opt = optim.RMSprop(net_params, lr=lr, momentum=momentum)
    elif opt_name == "Adamax":
        opt = optim.Adamax(net_params, lr=lr)
    elif opt_name == "Adagrad":
        opt = optim.Adagrad(net_params, lr=lr)
    elif opt_name == "Adadelta":
        opt = optim.Adadelta(net_params, lr=lr)
    elif opt_name == "Nadam":
        opt= optim.NAdam(net_params, lr=lr)
    else:
        sys.exit("Invalid optimizer")
    return opt

def map_loss_func(loss_name):
    if loss_name == "MSELoss":
        loss_func = torch.nn.MSELoss()
    elif loss_name == "SmoothL1Loss":
        loss_func = torch.nn.SmoothL1Loss()
    else:
        sys.exit("Invalid loss function")
    return loss_func