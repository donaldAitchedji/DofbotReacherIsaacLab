import cv2 as cv
import numpy as np
from detection import WasteDetector
from arm_module import RobotArm
from agent import Agent
from env import ReachDofbotEnv
import utils as ut
import time

# Initialisation
detector = WasteDetector()
agent = Agent(model=r"models\abs_angles\best_agent_a2c.pt")
my_arm = RobotArm()
env = ReachDofbotEnv(my_arm)
cap = cv.VideoCapture(1)
cap.set(cv.CAP_PROP_FPS, 30)


while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. On cherche l'objet
    pose_object, label = detector.get_position(frame)

    if pose_object is not None:

        x_base, y_base, z_base = pose_object[0], pose_object[1], pose_object[2]
        print(f"Position en base: x={x_base}, y={y_base}, z={z_base}")

        # On met à jour l'environnement avec la position cible
        env.set_pose_tgt(pose_object)
        pose_picking = env.pos_tgt
        print(f"Position de prise: x={pose_picking[0]}, y={pose_picking[1]}, z={pose_picking[2]}")
        dist = np.linalg.norm(pose_picking - my_arm.get_claw_pose())

        while dist > 0.02:
            # 2. On calcule l'état 
            state_list = env.get_state(pose_object)

            # 3. On demande à l'agent de choisir une action 
            angles = agent.select_action(state_list)
            print (f"Angles choisis: {angles}")
            dist = np.linalg.norm(pose_picking - my_arm.get_claw_pose())

        """
        # 4. On effectue la prise si on atteint la position de prise
        if angles:
                my_arm.move_to_waste(angles)
                my_arm.close_gripper()
                my_arm.lift_object()
                my_arm.drop_in_bin()
                time.sleep(1)
        """

    cv.imshow("Robot Cam", frame)
    if cv.waitKey(1) & 0xFF == ord('q'): break

cap.release()