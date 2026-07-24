import torch
import torch.nn as nn
import numpy as np
import utils as ut 
from utils import DELTA_ANGLES


class PolicyNet(nn.Module):
    def __init__(self, num_obs, num_actions):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(num_obs, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x):
        return self.net(x)  # mean action

        
class Agent: 
    def __init__(self, model, num_obs, num_act):
        self.model = PolicyNet(num_obs, num_act)
        self.model.load_state_dict(ut.load_weights(model))
        self.model.eval()
    

    def select_action(self, state_list):
        state = torch.FloatTensor(state_list).unsqueeze(0)
        with torch.no_grad(): 
            action = torch.clamp(self.model(state), -1.0, 1.0)
            action = action.squeeze(0).numpy()
        action = action*DELTA_ANGLES
        print("Action selected : ",action)
        return action