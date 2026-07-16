import cv2 as cv
import numpy as np
from detection import WasteDetector
from arm_module import RobotArm
from agent import Agent
from env import ReachDofbotEnv


# Initialisation
detector = WasteDetector()
num_obs = 7
num_act = 4
agent = Agent("models/delta_angles/best_agent_2.pt", num_obs, num_act)
my_arm = RobotArm()
env = ReachDofbotEnv(my_arm)
cap = cv.VideoCapture(1)
cap.set(cv.CAP_PROP_FPS, 30)


while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. On cherche l'objet
    pose_object = detector.get_object_position(frame)

    if pose_object is not None:

        x_base, y_base, z_base = pose_object[0], pose_object[1], pose_object[2]
        print(f"Position en base: x={x_base}, y={y_base}, z={z_base}")

        # On met à jour l'environnement avec la position cible
        env.set_pose_tgt(pose_object)
        dist = np.linalg.norm(env.pos_tgt - my_arm.get_claw_pose())

        while dist > 0.02:
            # 2. On calcule l'état 
            state_list = env.get_state(pose_object)

            # 3. On demande à l'agent de choisir une action 
            delta_angles = agent.select_action(state_list)
            print (f"Delta angles choisis: {delta_angles}")
            angles = my_arm.get_current_angles()[:4]
            print(f"Angles précédents : {angles}")
            angles = angles + delta_angles
            print(f"Angles à appliquer: {angles}")
            my_arm.move_to_waste(angles)
            dist = np.linalg.norm(env.pos_tgt - my_arm.get_claw_pose())

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
cv.destroyAllWindows()