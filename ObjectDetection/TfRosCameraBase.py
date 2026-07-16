import numpy as np
from scipy.stats import special_ortho_group

T_cam_to_base_link=np.array([0.002, 0.139, 0.153])

q_cam_to_base_link= np.array([0.983, -0.009, -0.001, -0.181])
def quaternion_to_rotation_matrix(q):
   q = np.array(q, dtype=float)
   q /= np.linalg.norm(q) # Normalize
   q1, q2, q3,q0 = q
   R = np.array([
       [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
       [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
       [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
   ])
   return R


def pos_in_base(R,t,pos_cam):
    return R@pos_cam + t

# Generate random unit quaternion
def random_quaternion():
    v = np.random.normal(size=4)
    return v / np.linalg.norm(v)

q = random_quaternion()
R = quaternion_to_rotation_matrix(q_cam_to_base_link)
print(R)

# Check orthogonality and determinant
print(np.allclose(R @ R.T, np.eye(3))) # Should be True
print(np.isclose(np.linalg.det(R), 1.0)) # Should be True

