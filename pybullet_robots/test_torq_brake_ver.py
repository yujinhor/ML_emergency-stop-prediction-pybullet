import pybullet as p
import pybullet_data
import time
import math
import random
import numpy as np
import pandas as pd

# --- 0. 시뮬레이션 상수 ---
PHYSICS_TIME_STEP = 1.0 / 240.0
NUM_EPISODES = 50
MAX_SIM_TIME = 1000.0 

# 시작 위치 (X = -11.0)
START_POS = [-11.0, -10.7, 0.5]  # 높이 0.5 (바닥에 박히지 않게 띄움)
START_YAW_DEG = 176              
START_YAW_RAD = math.radians(START_YAW_DEG)

# --- 1. 환경 설정 ---
def setup_environment():
    p.setGravity(0, 0, -10)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 바닥 plane
    plane_id = p.loadURDF("plane_implicit.urdf")
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.4, 0.4, 0.4, 1.0])
    
    # 트랙 + 벽
    track_objects = p.loadSDF("f10_racecar/meshes/barca_track1.sdf", globalScaling=1)
    #track_objects = p.loadSDF("f10_racecar/meshes/barca_track2.sdf", globalScaling=1)

    
    wall_id = track_objects[-1]
    track_ids = track_objects[:-1]

    # 초기 마찰 세팅
    p.changeDynamics(plane_id, -1, lateralFriction=1.0)
    for t_id in track_ids:
        p.changeDynamics(t_id, -1, lateralFriction=1.0)
    
    return plane_id, track_ids, wall_id

# --- 2. 차량 로드 (Constraint 포함) ---
def load_racecar(pos, yaw):
    quat = p.getQuaternionFromEuler([0, 0, yaw])
    car = p.loadURDF("f10_racecar/racecar_differential.urdf", pos, quat)
    
    # 생성 후 잠시 물리 안정화 대기
    for _ in range(20):
        p.stepSimulation()
        
    # 물리 제약조건 (한 번만 설정하면 계속 유지됨)
    c = p.createConstraint(car, 9, car, 11, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=1, maxForce=10000)

    c = p.createConstraint(car, 10, car, 13, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car, 9, car, 13, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car, 16, car, 18, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=1, maxForce=10000)

    c = p.createConstraint(car, 16, car, 19, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car, 17, car, 19, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car, 1, car, 18, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)

    c = p.createConstraint(car, 3, car, 19, jointType=p.JOINT_GEAR, jointAxis=[0,1,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)
    
    return car

# --- 3. 랜덤 파라미터 ---
def get_random_conditions():
    target_friction = random.choice([1.0, 0.5, 0.1])
    target_speed = random.uniform(60,100)
    trigger_dist = random.uniform(0.1, 2.0)
    mass = random.uniform(1.0, 10.0)
    brake_torque = 100.0
    
    return {
        "friction": target_friction,
        "target_speed": target_speed,
        "trigger_dist": trigger_dist,
        "mass": mass,
        "brake_torque": brake_torque
    }

# --- 4. 메인 실행 ---
if __name__ == "__main__":
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    # 1. 환경 로드 (한 번만 수행)
    plane_id, track_ids, wall_id = setup_environment()
    
    wall_pos_abs, _ = p.getBasePositionAndOrientation(wall_id)
    wall_x = wall_pos_abs[0]
    
    # 2. 차량 로드 (한 번만 수행 - 중요!)
    car_id = load_racecar(START_POS, START_YAW_RAD)
    
    summary_rows = []
    all_step_rows = []
    
    print(f"--- 데이터 수집 시작 (Reset Position Method) ---")
    
    wheels = [8, 15]   
    steering = [0, 2] 

    for ep in range(NUM_EPISODES):
        # ==================================================
        # [중요] 에피소드 초기화 (차량을 지우지 않고 위치만 리셋)
        # ==================================================
        
        # 1. 차량 위치/자세 리셋
        p.resetBasePositionAndOrientation(
            car_id, 
            START_POS, 
            p.getQuaternionFromEuler([0, 0, START_YAW_RAD])
        )
        
        # 2. 차량 속도 리셋 (정지 상태로)
        p.resetBaseVelocity(car_id, [0, 0, 0], [0, 0, 0])
        
        # 3. 환경 마찰력 리셋 (이전 에피소드에서 아이스반으로 변했을 수 있으므로 복구)
        p.changeDynamics(plane_id, -1, lateralFriction=1.0)
        for t_id in track_ids:
            p.changeDynamics(t_id, -1, lateralFriction=1.0)
            
        # 4. 랜덤 조건 생성 및 적용
        cond = get_random_conditions()
        
        # 차량 질량 적용
        p.changeDynamics(car_id, -1, mass=cond['mass'])
        # 차량 휠 마찰 리셋 (혹시 모르니 1.0으로)
        for i in range(p.getNumJoints(car_id)):
            p.changeDynamics(car_id, i, lateralFriction=1.0)
        p.changeDynamics(car_id, -1, lateralFriction=1.0)
        
        # 모터 제어 초기화 (이전 에피소드의 토크 제거)
        for w in wheels:
            p.setJointMotorControl2(car_id, w, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            
        # 잠시 대기 (물리 엔진이 리셋된 위치를 인지하도록)
        for _ in range(10):
            p.stepSimulation()
            
        print(
            f"[{ep+1}/{NUM_EPISODES}] "
            f"m:{cond['mass']:.1f}kg, v_cmd:{cond['target_speed']:.1f}, "
            f"μ_black:{cond['friction']}, trigger:{cond['trigger_dist']:.2f}m... ",
            end=""
        )
        
        sim_time = 0.0
        is_failure = 0
        stop_distance = 0.0
        max_speed_achieved = 0.0
        is_braking_active = False
        episode_steps = []
        
        speed_at_trigger = None
        time_at_trigger = None
        dist_at_trigger = None
        time_to_stop = None
        
        while True:
            car_pos, _ = p.getBasePositionAndOrientation(car_id)
            car_vel, _ = p.getBaseVelocity(car_id)
            speed = np.linalg.norm(car_vel)
            
            if speed > max_speed_achieved:
                max_speed_achieved = speed
            
            dist_to_wall = abs(car_pos[0] - wall_x)
            
            # --- 트리거 & 블랙아이스 로직 ---
            if dist_to_wall <= cond['trigger_dist']:
                if not is_braking_active:
                    speed_at_trigger = speed
                    time_at_trigger = sim_time
                    dist_at_trigger = dist_to_wall
                    
                    # 환경 마찰력 변경
                    ground_ids = [plane_id] + list(track_ids)
                    for gid in ground_ids:
                        p.changeDynamics(gid, -1, lateralFriction=cond['friction'])
                is_braking_active = True
            
            # --- 조향 로직 ---
            if dist_to_wall <= 1.5:
                steer_cmd = -1.0
            else:
                steer_cmd = 0.0

            # --- 구동 / 제동 제어 ---
            if is_braking_active:
                brake_torque_cmd = cond['brake_torque']
                for w in wheels:
                    p.setJointMotorControl2(
                        car_id, w, controlMode=p.TORQUE_CONTROL, force=-brake_torque_cmd
                    )
            else:
                brake_torque_cmd = 0.0
                for w in wheels:
                    p.setJointMotorControl2(
                        car_id, w, controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=cond['target_speed'], force=200
                    )
            
            for s in steering:
                p.setJointMotorControl2(
                    car_id, s, controlMode=p.POSITION_CONTROL, targetPosition=steer_cmd
                )
                
            current_friction = cond['friction'] if is_braking_active else 1.0

            # 로그 기록
            step_row = {
                "episode": ep,
                "time": sim_time,
                "x": car_pos[0],
                "y": car_pos[1],
                "z": car_pos[2],
                "speed": speed,
                "dist_to_wall": dist_to_wall,
                "current_friction": current_friction,
                "is_braking": int(is_braking_active),
                "brake_torque_cmd": brake_torque_cmd,
                "steer_cmd": steer_cmd,
                "mass_kg": cond['mass'],
                "friction": cond['friction'],
                "init_speed_cmd": cond['target_speed'],
                "trigger_dist_m": cond['trigger_dist'],
                "base_brake_torque": cond['brake_torque'],
            }
            episode_steps.append(step_row)
            
            p.stepSimulation()
            sim_time += PHYSICS_TIME_STEP
            
            p.resetDebugVisualizerCamera(5.0, 90, -40, car_pos)
            
            # 이탈 확인
            if car_pos[2] < -1.0 or car_pos[2] > 2.0:
                print(" -> ⚠️ 오류: 차량 이탈")
                is_failure = -1
                break

            # 충돌 감지
            is_collision = False
            if len(p.getContactPoints(car_id, wall_id)) > 0:
                is_collision = True
            for t_id in track_ids:
                for c in p.getContactPoints(car_id, t_id):
                    if abs(c[7][2]) < 0.7:
                        is_collision = True
                        break
                if is_collision: break
            
            if is_collision:
                is_failure = 1
                print(f" -> 💥 충돌!")
                break
            
            # 정지 성공
            if is_braking_active and speed < 0.05:
                if max_speed_achieved < 0.5:
                    print(" -> ⚠️ 출발 실패")
                    is_failure = -1
                    break
                is_failure = 0
                stop_distance = dist_to_wall
                time_to_stop = sim_time - time_at_trigger if time_at_trigger else 0
                print(f" -> ✅ 정지 성공 ({stop_distance:.2f}m)")
                break
            
            if sim_time > MAX_SIM_TIME:
                print(" -> ⏰ 시간 초과")
                is_failure = -1
                break
        
        # 데이터 저장
        if is_failure != -1:
            summary_row = {
                "episode": ep,
                "mass_kg": cond['mass'],
                "friction": cond['friction'],
                "init_speed_cmd": cond['target_speed'],
                "trigger_dist_m": cond['trigger_dist'],
                "brake_torque": cond['brake_torque'],
                "result_is_failure": is_failure,
                "final_dist_to_wall": stop_distance if is_failure == 0 else 0.0,
                "max_speed_achieved": max_speed_achieved,
                "speed_at_trigger": speed_at_trigger if speed_at_trigger else 0.0,
                "time_at_trigger": time_at_trigger if time_at_trigger else 0.0,
                "dist_at_trigger": dist_at_trigger if dist_at_trigger else 0.0,
                "time_to_stop": time_to_stop if time_to_stop else 0.0,
            }
            summary_rows.append(summary_row)
            for row in episode_steps:
                row["result_is_failure"] = is_failure
            all_step_rows.extend(episode_steps)
            
    p.disconnect()
    
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv("black_ice_data.csv", index=False)
        print("\n[완료] 데이터 저장됨.")
        pd.DataFrame(all_step_rows).to_csv("black_ice_steps.csv", index=False)
