import torch
import torch.nn as nn
from typing import TypeVar


class NetworkType(nn.Module):
    def reset_parameters(self):
        raise NotImplementedError("You shoulnd't see this.")

    def reset_noise(self):
        if hasattr(self.net, 'reset_noise'):
            self.net.reset_noise()

    def act(self, *args):
        with torch.no_grad():
            self.eval()
            x = self.forward(*args)
            self.train()
            return x


NetworkTypeClass = TypeVar("NetworkTypeClass", bound=NetworkType)
