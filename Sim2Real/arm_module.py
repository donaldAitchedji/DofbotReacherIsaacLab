#!/usr/bin/env python3
import sys
import time
sys.path.append("/home/pi/Documents/Arm_lib-1.0.0/Arm_Lib")
from math import pi, cos, sin
from Arm_Lib import Arm_Device


class RobotArm:
    def __init__(self):
        # Initialisation matériel
        self.arm = Arm_Device()

        # Constantes
        self.RA2DE = 180 / pi
        self.DE2RA = pi / 180
        self.gripper_closed = 135
        self.gripper_open = 49

        # Trajectoire de la corbeille
        self.bin = [0,102,5,14,91, self.gripper_closed]

        self.go_home()
        self.prev_angles = [90, 99, 3, 9, 91, 49]
    

    # ================= COMMANDES BAS NIVEAU =================
    def move_joints(self, j1, j2, j3, j4, j5=91, j6=50, t=1000):
        self.arm.Arm_serial_servo_write6(j1, j2, j3, j4, j5, j6, t)
        time.sleep(t / 1000)

    def open_gripper(self):
        self.arm.Arm_serial_servo_write(6, self.gripper_open, 500)
        time.sleep(0.5)

    def close_gripper(self):
        self.arm.Arm_serial_servo_write(6, self.gripper_closed, 500)
        time.sleep(0.5)

    # ================= SCÉNARIOS =================
    def go_home(self):
        self.move_joints(90, 99, 3, 9, 91, self.gripper_open)
        time.sleep(1)

    def move_to_waste(self, angles):
        j1, j2, j3, j4 = angles
        self.move_joints(j1, j2, j3, j4)

    def lift_object(self):
        self.arm.Arm_serial_servo_write(2, 83, 1000)
        time.sleep(1)

    def drop_in_bin(self):
        traj = self.bin
        self.move_joints(traj[0], traj[1], traj[2], traj[3], traj[4], traj[5])
        self.open_gripper()
        self.go_home()
        time.sleep(1)

    def get_claw_pose(self) :
        angles = self.get_current_angles()
            
        alpha = angles[1] * self.DE2RA
        beta = angles[2] * self.DE2RA
        gamma = angles[3] * self.DE2RA
        theta = angles[0] * self.DE2RA

        l1 = 0.1075 # l1 = O0O1+O1O2
        l2 = 0.08285 # l2 = O2O3 = O3O4
        l3 = 0.17385 # l3 représente ici l4 + l5

        x = cos(theta)*(l2*(cos(alpha) + sin(alpha + beta)) - l3 * cos(alpha + beta + gamma))
        y = sin(theta)*(l2*(cos(alpha) + sin(alpha + beta)) - l3 * cos(alpha + beta + gamma))
        z = l1 + l2*(sin(alpha) - cos(alpha + beta)) - l3 * sin(alpha + beta + gamma)

        return x, y, z

    def get_current_angles(self, max_retries=1, retry_delay=0.05): #0.02):
        angles = []
        for i in range(1, 7):
            angle = None
            if i != 5 and i!= 6: #les 4 premiers
                for attempt in range(max_retries + 1):
                    angle = self.arm.Arm_serial_servo_read(i)
                    if angle is not None:
                        break
            else: # on teste une seule fois  pour la pince et le wrist twist
                angle = self.arm.Arm_serial_servo_read(i)
                
            if angle is None:
                print(f"[WARN] Lecture échouée pour le servo {i} après une tentatives — repli sur prev_angles[{i-1}]")
                angle = self.prev_angles[i - 1]

            angles.append(angle)

        self.prev_angles = angles
        return angles
#arm = RobotArm()
#print(arm.prev_angles)
#arm.move_joints(arm.prev_angles[0]+3,arm.prev_angles[1]+3,arm.prev_angles[2]+3,arm.prev_angles[3]+3)
#print(arm.get_current_angles())
#print(arm.prev_angles)