import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(obs_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
