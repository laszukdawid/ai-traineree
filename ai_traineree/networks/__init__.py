from typing import TypeVar

import torch
import torch.nn as nn


class NetworkType(nn.Module):
    def reset_parameters(self):
        pass

    def reset_noise(self):
        if hasattr(self.net, "reset_noise"):
            self.net.reset_noise()

    @torch.no_grad()
    def act(self, *args):
        self.eval()
        x = self.forward(*args)
        self.train()
        return x


NetworkTypeClass = TypeVar("NetworkTypeClass", bound=NetworkType)
