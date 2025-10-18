import torch
import torch.nn as nn
import torch.nn.functional as F

class PPONetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(PPONetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)
        
    def forward(self, x):
        shared_out = self.shared(x)
        return self.actor(shared_out), self.critic(shared_out)