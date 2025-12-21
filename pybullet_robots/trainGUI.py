import pybullet as p
import pybullet_data
import time
import math
import random
import numpy as np
import pandas as pd
import os 

# 시뮬레이션 상수
PHYSICS_TIME_STEP = 1.0 / 240.0
NUM_EPISODES = 10000      
MAX_SIM_TIME = 1000.0 

TARGET_LOG_HZ = 10
LOG_INTERVAL_STEPS = int((1.0 / PHYSICS_TIME_STEP) / TARGET_LOG_HZ)

# 난수 설정
GLOBAL_SEED = 42
# 훈련/평가 데이터 비율
TRAIN_RATIO = 0.8 


USE_GUI = True
FOLLOW_CAMERA = True  
SLOW_IN_GUI = False   

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

# 공기 저항 상수
DRAG_COEFF = 0.82
FRONTAL_AREA = 0.0532


START_POS = [-11.0, -10.7, 0.5]
START_YAW_DEG = 176
START_YAW_RAD = math.radians(START_YAW_DEG)


def update_follow_camera(car_id):
    car_pos, car_orn = p.getBasePositionAndOrientation(car_id)
    
    target_pos = [car_pos[0], car_pos[1], car_pos[2] + 0.5]

    camera_distance = 6.0  
    camera_yaw = 50         
    camera_pitch = -25      

    p.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=camera_yaw,
        cameraPitch=camera_pitch,
        cameraTargetPosition=target_pos
    )
def setup_environment():
    p.setGravity(0, 0, -10)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    plane_id = p.loadURDF("plane_implicit.urdf")
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.4, 0.4, 0.4, 1.0])
    
    track_objects = p.loadSDF("f10_racecar/meshes/barca_track.sdf", globalScaling=1)
    
    wall_id = track_objects[-1]
    track_ids = track_objects[:-1]

    p.changeDynamics(plane_id, -1, lateralFriction=1.0)
    for t_id in track_ids:
        p.changeDynamics(t_id, -1, lateralFriction=1.0)
    
    return plane_id, track_ids, wall_id

def load_racecar(pos, yaw):
    quat = p.getQuaternionFromEuler([0, 0, yaw])
    car = p.loadURDF("f10_racecar/racecar_differential.urdf", pos, quat)
    
    for _ in range(20):
        p.stepSimulation()
        
    # 차량 물리 제약
    c = p.createConstraint(car, 9, car, 11, jointType=p.JOINT_GEAR,
                           jointAxis=[0,1,0], parentFramePosition=[0,0,0],
                           childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=1, maxForce=10000)

    c = p.createConstraint(car, 10, car, 13, jointType=p.JOINT_GEAR,
                           jointAxis=[0,1,0], parentFramePosition=[0,0,0],
                           childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car, 9, car, 13, jointType=p.JOINT_GEAR,
                           jointAxis=[0,1,0], parentFramePosition=[0,0,0],
                           childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car, 16, car, 18, jointType=p.JOINT_GEAR,
                           jointAxis=[0,1,0], parentFramePosition=[0,0,0],
                           childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=1, maxForce=10000)

    c = p.createConstraint(car, 16, car, 19, jointType=p.JOINT_GEAR,
                           jointAxis=[0,1,0], parentFramePosition=[0,0,0],
                           childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car, 17, car, 19, jointType=p.JOINT_GEAR,
                           jointAxis=[0,1,0], parentFramePosition=[0,0,0],
                           childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car, 1, car, 18, jointType=p.JOINT_GEAR,
                           jointAxis=[0,1,0], parentFramePosition=[0,0,0],
                           childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)

    c = p.createConstraint(car, 3, car, 19, jointType=p.JOINT_GEAR,
                           jointAxis=[0,1,0], parentFramePosition=[0,0,0],
                           childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)
    
    return car

# 물리 변수 설정
def get_random_conditions():
    target_friction = random.choice([1.0, 0.5, 0.1])
    target_speed = random.uniform(60, 80)
    trigger_dist = random.uniform(0.05, 0.30)
    mass = random.uniform(4.0, 8.0)
    brake_torque = 2.0
    
    return {
        "friction": target_friction,
        "target_speed": target_speed,
        "trigger_dist": trigger_dist,
        "mass": mass,
        "brake_torque": brake_torque
    }

def get_air_density(friction):
    if friction >= 0.9:
        return 1.225
    elif friction >= 0.4:
        return 1.150
    else:
        return 1.350

# 공기 저항 함수
def apply_drag_force(car_id, air_density):
    vel_vec, _ = p.getBaseVelocity(car_id)
    speed = np.linalg.norm(vel_vec)
    
    if speed < 0.01:
        return 0.0

    drag_magnitude = 0.5 * air_density * DRAG_COEFF * FRONTAL_AREA * (speed ** 2)
    
    drag_force_vec = [
        -drag_magnitude * (vel_vec[0] / speed),
        -drag_magnitude * (vel_vec[1] / speed),
        -drag_magnitude * (vel_vec[2] / speed)
    ]
    
    p.applyExternalForce(car_id, -1, drag_force_vec, [0, 0, 0], p.LINK_FRAME)
    
    return drag_magnitude

if __name__ == "__main__":
    set_global_seed(GLOBAL_SEED)
    print(f"[INFO] Global random seed fixed to {GLOBAL_SEED}")
    print(f"[INFO] Data collection frequency set to {TARGET_LOG_HZ}Hz (Log every {LOG_INTERVAL_STEPS} steps)")

    for fname in ["train_summary1.csv", "train_steps1.csv",
                  "val_summary1.csv", "val_steps1.csv"]:
        if os.path.exists(fname):
            os.remove(fname)

    # PyBullet 연결
    if USE_GUI:
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.setRealTimeSimulation(0)
        p.resetDebugVisualizerCamera(
            cameraDistance=8.0,
            cameraYaw=50,
            cameraPitch=-35,
            cameraTargetPosition=[0, 0, 0]
        )
        print("[INFO] PyBullet connected with GUI mode")
    else:
        p.connect(p.DIRECT)
        print("[INFO] PyBullet connected with DIRECT mode (no GUI)")
    
    plane_id, track_ids, wall_id = setup_environment()
    wall_pos_abs, _ = p.getBasePositionAndOrientation(wall_id)
    wall_x = wall_pos_abs[0]
    
    car_id = load_racecar(START_POS, START_YAW_RAD)
    
    print(f"\n--- 데이터 수집 시작 (총 {NUM_EPISODES}회, {'GUI' if USE_GUI else 'GUI 없음'}) ---")
    print(f"{'EP':<5} | {'Mass(kg)':<8} | {'Vel(cmd)':<8} | {'Fric(μ)':<7} | "
          f"{'TrigDist(m)':<11} | {'MaxDrag(N)':<10} | {'Result':<10} | {'FinalDist(m)':<10}")
    print("-" * 105)
    
    wheels = [8, 15]   
    steering = [0, 2]

    train_success_eps = 0
    val_success_eps = 0

    for ep in range(NUM_EPISODES):
        p.resetBasePositionAndOrientation(
            car_id, START_POS, p.getQuaternionFromEuler([0, 0, START_YAW_RAD])
        )
        p.resetBaseVelocity(car_id, [0, 0, 0], [0, 0, 0])
        
        p.changeDynamics(plane_id, -1, lateralFriction=1.0)
        for t_id in track_ids:
            p.changeDynamics(t_id, -1, lateralFriction=1.0)
            
        cond = get_random_conditions()
        current_air_density = get_air_density(cond['friction'])
        
        p.changeDynamics(car_id, -1, mass=cond['mass'])
        for i in range(p.getNumJoints(car_id)):
            p.changeDynamics(car_id, i, lateralFriction=1.0)
        p.changeDynamics(car_id, -1, lateralFriction=1.0)
        
        for w in wheels:
            p.setJointMotorControl2(car_id, w, p.VELOCITY_CONTROL,
                                    targetVelocity=0, force=0)
            
        for _ in range(10):
            p.stepSimulation()
            if USE_GUI and SLOW_IN_GUI:
                time.sleep(PHYSICS_TIME_STEP)
            
        print(f"[{ep+1:03d}/{NUM_EPISODES}] | {cond['mass']:<8.1f} | "
              f"{cond['target_speed']:<8.1f} | {cond['friction']:<7.1f} | "
              f"{cond['trigger_dist']:<11.2f} | ", end="", flush=True)
        
        sim_time = 0.0
        step_counter = 0
        is_failure = 0
        stop_distance = 0.0
        max_speed_achieved = 0.0
        is_braking_active = False
        max_drag_val = 0.0
        
        speed_at_trigger = None
        time_at_trigger = None
        episode_steps = []

        while True:
            car_pos, _ = p.getBasePositionAndOrientation(car_id)
            if USE_GUI and FOLLOW_CAMERA:
            	update_follow_camera(car_id)
            car_vel, _ = p.getBaseVelocity(car_id)
            speed = np.linalg.norm(car_vel)
            
            if speed > max_speed_achieved:
                max_speed_achieved = speed
            
            dist_to_wall = abs(car_pos[0] - wall_x)
            
            # 공기 저항 설정
            drag_force_mag = apply_drag_force(car_id, current_air_density)
            if drag_force_mag > max_drag_val:
                max_drag_val = drag_force_mag
            
            # 정지 트리거
            if dist_to_wall <= cond['trigger_dist']:
                if not is_braking_active:
                    speed_at_trigger = speed
                    time_at_trigger = sim_time
                    
                    ground_ids = [plane_id] + list(track_ids)
                    for gid in ground_ids:
                        p.changeDynamics(gid, -1, lateralFriction=cond['friction'])
                is_braking_active = True
            
            if dist_to_wall <= 1.5: 
                steer_cmd = -1.0
            else: 
                steer_cmd = 0.0

            # 제동 속도 제어, 브레이크 토크 제어
            if is_braking_active:
                brake_torque_val = cond['brake_torque']
                for w in wheels:
                    w_vel = p.getJointState(car_id, w)[1]
                    if w_vel > 0.5:
                        p.setJointMotorControl2(
                            car_id, w,
                            controlMode=p.TORQUE_CONTROL,
                            force=-brake_torque_val
                        )
                    else:
                        p.setJointMotorControl2(
                            car_id, w,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=0,
                            force=100
                        )
            else:
                for w in wheels:
                    p.setJointMotorControl2(
                        car_id, w,
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=cond['target_speed'],
                        force=20
                    )
            
            for s in steering:
                p.setJointMotorControl2(
                    car_id, s,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=steer_cmd
                )
            
            # 10Hz로 저장
            if step_counter % LOG_INTERVAL_STEPS == 0:
                step_row = {
                    "episode": ep,
                    "time": sim_time,
                    "speed": speed,
                    "dist_to_wall": dist_to_wall,
                    "drag_force_N": drag_force_mag,
                    "is_braking": int(is_braking_active),
                    "friction": cond['friction'],
                    "trigger_dist_m": cond['trigger_dist'],
                    "mass": cond['mass']
                }
                episode_steps.append(step_row)
            
            p.stepSimulation()
            if USE_GUI and SLOW_IN_GUI:
                time.sleep(PHYSICS_TIME_STEP)

            
            sim_time += PHYSICS_TIME_STEP
            step_counter += 1
            
            # 이탈 확인
            if car_pos[2] < -1.0 or car_pos[2] > 2.0:
                is_failure = -1
                break

            # 충돌 확인
            is_collision = False
            if len(p.getContactPoints(car_id, wall_id)) > 0: 
                is_collision = True
            for t_id in track_ids:
                if is_collision:
                    break
                for c in p.getContactPoints(car_id, t_id):
                    if abs(c[7][2]) < 0.7:
                        is_collision = True
                        break
            
            if is_collision:
                is_failure = 1
                break
            
            # 정지 성공
            if is_braking_active and speed < 0.05:
                if max_speed_achieved < 0.5:
                    is_failure = -1  # 출발 실패
                    break
                is_failure = 0
                stop_distance = dist_to_wall
                break
            
            if sim_time > MAX_SIM_TIME:
                is_failure = -1  # 시간 초과
                break
        
        if is_failure == 0: result_str = "Success"
        elif is_failure == 1: result_str = "Crash"
        else: result_str = "Error"
        
        print(f"{max_drag_val:<10.2f} | {result_str:<10} | {stop_distance:<10.2f}")

        if is_failure == -1:
            continue

        is_train = random.random() < TRAIN_RATIO

        summary_row = {
            "episode": ep,
            "mass_kg": cond['mass'],
            "friction_cond": cond['friction'],
            "air_density": current_air_density,
            "init_speed_cmd": cond['target_speed'],
            "trigger_dist_m": cond['trigger_dist'],
            "brake_torque": cond['brake_torque'],
            "result_is_failure": is_failure,
            "final_dist_to_wall": stop_distance if is_failure == 0 else 0.0,
            "max_drag_force": max_drag_val
        }

        for row in episode_steps:
            row["result_is_failure"] = is_failure

        df_summary_ep = pd.DataFrame([summary_row])
        df_steps_ep = pd.DataFrame(episode_steps)

        # CSV 저장
        if is_train:
            train_success_eps += 1
            df_summary_ep.to_csv(
                "train_summary1.csv",
                mode="a",
                header=not os.path.exists("train_summary1.csv"),
                index=False
            )
            df_steps_ep.to_csv(
                "train_steps1.csv",
                mode="a",
                header=not os.path.exists("train_steps1.csv"),
                index=False
            )
        else:
            val_success_eps += 1
            df_summary_ep.to_csv(
                "val_summary1.csv",
                mode="a",
                header=not os.path.exists("val_summary1.csv"),
                index=False
            )
            df_steps_ep.to_csv(
                "val_steps1.csv",
                mode="a",
                header=not os.path.exists("val_steps1.csv"),
                index=False
            )

    p.disconnect()

    print("\n--- 요약 ---")
    print(f"총 에피소드 수               : {NUM_EPISODES}")
    print(f"성공/충돌 에피소드 (train)   : {train_success_eps}")
    print(f"성공/충돌 에피소드 (val)     : {val_success_eps}")

