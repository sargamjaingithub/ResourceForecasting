import torch
import torch.nn as nn
from torch.functional import F


def select_loss_function(config, device):
    return eval(config['name'])(config, device)


class Default:
    def __init__(self, config, device):
        # TODO refactor to do it generic
        super().__init__()
        self.criterion_name = config['criterion']
        criterion = config['criterion']
        if criterion == 'binary_cross_entropy':
            bce_loss = nn.BCELoss()
            sigmoid = nn.Sigmoid()
            self.criterion = lambda output, target: bce_loss(sigmoid(output), sigmoid(target))
        elif criterion == 'negative_log_likelihood':
            self.criterion = nn.NLLLoss()
        elif criterion == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
            # self.criterion = lambda output, target: loss(output.float(), target.float())
        elif criterion == 'mse':
            self.criterion = nn.MSELoss()
        elif criterion == 'exponential_mse':
            self.criterion = lambda input, target: torch.sum((input - target) ** config['rate']).mean()
        elif criterion == 'exponential_mse_sigmoid':
            self.criterion = lambda input, target: torch.sum(nn.Sigmoid()((input - target) ** config['rate'])).mean()
        elif criterion == 'weighted_mse':
            weights = torch.flip(torch.arange(config['prediction_interval']), [0])
            weights = (weights - weights.min()) / (weights.max() - weights.min())
            weights = weights.float().to(device)
            self.criterion = lambda input, target: torch.sum(weights * (input - target) ** 2).mean()
        elif criterion == 'weighted_mse_sigmoid':
            weights = torch.flip(torch.arange(config['prediction_interval']), [0])
            weights = (weights - weights.min()) / (weights.max() - weights.min())
            weights = weights.float().to(device)
            self.criterion = lambda input, target: torch.sum(nn.Sigmoid()(weights * (input - target) ** 2)).mean()
        else:
            raise Exception('Loss function criterion not recognized')

    def run(self, output, target):
        return self.criterion(output, target)


class CdCE:
    def __init__(self, config, device):
        super().__init__()
        # from here https://github.com/haitongli/knowledge-distillation-pytorch/blob/master/model/net.py
        self.alpha = config['alpha']
        self.T = config['temperature']
        self.kld_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        self.device = device

    def run(self, output, target):
        real_target, weighted_target = target

        kld_output = self.kld_loss(F.log_softmax(output / self.T, dim=1), F.softmax(weighted_target / self.T, dim=1))
        ce_output = self.ce_loss(output, real_target)

        return kld_output * (self.alpha * self.T * self.T) + ce_output * (1. - self.alpha)
