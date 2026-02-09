import torch
from torch.optim import Optimizer
import math


def get_optimizer(model_parameters, config):
    """
    Get optimizer based on config

    Args:
        model_parameters: model parameters to optimize
        config: configuration dict with optimizer settings

    Returns:
        optimizer instance
    """
    optimizer_name = config.get('optimizer', 'adam').lower()
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 0.0)

    if optimizer_name == 'adam':
        return torch.optim.Adam(
            model_parameters,
            lr=lr,
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.999)),
            eps=config.get('eps', 1e-8),
            weight_decay=weight_decay
        )

    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(
            model_parameters,
            lr=lr,
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.999)),
            eps=config.get('eps', 1e-8),
            weight_decay=weight_decay
        )

    elif optimizer_name == 'sgd': # Stochastic Gradient Descent
        return torch.optim.SGD(
            model_parameters,
            lr=lr,
            momentum=config.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=config.get('nesterov', False)
        ) # Nesterov Momentum for faster optimizing

    elif optimizer_name == 'rmsprop':
        return torch.optim.RMSprop(
            model_parameters,
            lr=lr,
            alpha=config.get('alpha', 0.99),
            eps=config.get('eps', 1e-8),
            weight_decay=weight_decay,
            momentum=config.get('momentum', 0.0)
        )

    elif optimizer_name == 'adagrad':
        return torch.optim.Adagrad(
            model_parameters,
            lr=lr,
            lr_decay=config.get('lr_decay', 0.0),
            weight_decay=weight_decay,
            eps=config.get('eps', 1e-10)
        )

    elif optimizer_name == 'adadelta':
        return torch.optim.Adadelta(
            model_parameters,
            lr=lr,
            rho=config.get('rho', 0.9),
            eps=config.get('eps', 1e-6),
            weight_decay=weight_decay
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")