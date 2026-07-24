import numpy as np
import torch
import copy as cp

DELTA_ANGLES = np.full(4,3,dtype= np.float32)

def normaliser(val, max_val):
  val_list = np.array([v for v in val])
  val_list = val_list / max_val
  val_list = np.array([ min(max(-1.0, v), 1.0) for v in val_list], dtype= np.float32)
  return val_list

def denormaliser(val, max_val):
  val_list = np.array([v for v in val])
  val_list = val_list * max_val
  val_list = np.array([ min(max(-max_val, v), max_val) for v in val_list], dtype= int)
  return val_list

def distance_axes_tgt_cur(pos_cur, pos_tgt):
  x_tgt, y_tgt, z_tgt = pos_tgt[0], pos_tgt[1], pos_tgt[2]
  x_cur, y_cur, z_cur = pos_cur[0], pos_cur[1], pos_cur[2]
  return np.array([x_tgt - x_cur, y_tgt - y_cur, z_tgt - z_cur])

def get_picking_pose(pos_tgt, offset=0.05):
    x_tgt, y_tgt, z_tgt = pos_tgt[0], pos_tgt[1], pos_tgt[2]

    if x_tgt >= 0:
        x_tgt = x_tgt - offset
    else:
        x_tgt = x_tgt + offset
    
    y_tgt = y_tgt - offset
    z_tgt = z_tgt + offset

    return np.array([x_tgt, y_tgt, z_tgt])

def load_weights(model):
    model_dict = torch.load(model,map_location= "cpu")
    dict_without_std = {}
    keys = model_dict["policy"].keys()
    for key in keys:
        if key == "log_std_parameter":
            continue
        dict_without_std[key] = model_dict["policy"][key]
    return dict_without_std
    
def clip_angles(angles):
    angles_clipped = cp.deepcopy(angles)
    angles_clipped[1:3] = np.clip(angles_clipped[1:3],0,180)
    angles_clipped[0],angles_clipped[3] = min(max(0,angles_clipped[0]),180),min(max(0,angles_clipped[3]),180)
    return angles_clipped

#angles = np.array([-1,99,89,189])
#angles = clip_angles(angles)
#print(angles)