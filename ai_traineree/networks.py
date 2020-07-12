import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed=np.random.random()):
        super(QNetwork, self).__init__()

        hl: int = 256
        self.fc1 = nn.Linear(state_size, hl)
        self.fc2 = nn.Linear(hl, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x