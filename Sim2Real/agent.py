import torch
import torch.nn as nn

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
        print("Action selected : ",action)
        #action = action*DELTA_ANGLES # on ramène l'action à un delta d'angle (-3 , 3)
        #action = ut.denormaliser(action, 90)
        action = action + DELTA_ANGLES # on pousse ces deltas d'angles pour qu'ils soient compris entre 0 et 6
        
        return action