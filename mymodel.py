import torch
import torch.nn as nn
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super(QNetwork, self).__init__()
        # Một mạng nơ-ron đơn giản với 2 lớp fully connected
        self.fc1 = nn.Linear(np.prod(observation_shape), 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_shape)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)  # Flatten the observation
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)