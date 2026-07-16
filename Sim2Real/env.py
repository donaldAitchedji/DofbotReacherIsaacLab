import numpy as np
import copy as cp
from arm_module import RobotArm
import utils as ut


class ReachDofbotEnv:
    def __init__(self, arm : RobotArm , pos_tgt = np.array([0.1, 0.0, 0.1])):
        
        # le bras robotique
        self.arm = arm
        # position de la zone de prise  
        self.pos_tgt = pos_tgt
    
    def set_pose_tgt(self, pos_tgt):
       self.pos_tgt = pos_tgt
        
    def get_state(self, pos_tgt):
        # position de la zone de prise  
        self.pos_tgt =pos_tgt
        
        # position actuelle du robot
        pos_dofbot = self.arm.get_claw_pose()  

        # différence suivant chaque axe
        delta_pose = ut.distance_axes_tgt_cur(pos_dofbot, self.pos_tgt)

        angles = np.array([self.arm.get_current_angles()[0],self.arm.get_current_angles()[1],  self.arm.get_current_angles()[2],self.arm.get_current_angles()[3]])

        angles = ut.normaliser(angles - 90, 90)

        state = np.concatenate([delta_pose, angles])
        
        return state




