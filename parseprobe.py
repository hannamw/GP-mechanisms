import torch
from torch import nn
import torch.functional as F

class ParseProbe(nn.Module):
    def __init__(self, model_dim=512, probe_dim=1024):
        super(ParseProbe, self).__init__()
        self.model_dim = model_dim
        self.probe_dim = probe_dim
        self.root = nn.Parameter(data=torch.zeros(self.model_dim))
        layers = [nn.Linear(2 * self.model_dim, self.probe_dim), nn.ReLU(), nn.Linear(self.probe_dim, self.probe_dim), nn.ReLU(), nn.Linear(self.probe_dim, 3)]
        self.transform = nn.Sequential(*layers)

    def forward(self, x):
        return self.transform(x)

    def from_sap(self, state_dict):
        old_names = ['root', 'transform.1.0.weight', 'transform.1.0.bias', 'transform.2.0.weight', 'transform.2.0.bias', 'transform.3.weight', 'transform.3.bias']
        new_names = ['root', 'transform.0.weight', 'transform.0.bias', 'transform.2.weight', 'transform.2.bias', 'transform.4.weight', 'transform.4.bias']
        new_state_dict = {new_name: state_dict[old_name] for old_name, new_name in zip(old_names, new_names)}
        self.load_state_dict(new_state_dict)

# Load like this:
# probe = ParseProbe()
# probe.load_state_dict(torch.load('layer2.pt'))