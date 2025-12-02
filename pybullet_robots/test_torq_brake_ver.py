import pybullet as p
import pybullet_data
import time
import math
import random
import numpy as np
import pandas as pd

# --- 0. 시뮬레이션 상수 ---
PHYSICS_TIME_STEP = 1.0 / 240.0
NUM_EPISODES = 500
MAX_SIM_TIME = 1000.0 

# [추가] 데이터 수집 주기를 위한 설정
TARGET_LOG_HZ = 10  # 목표 데이터 수집 주파수 (10Hz)
# 240Hz 물리 스텝 중 몇 번마다 저장할지 계산 (240 / 10 = 24스텝마다 저장)
LOG_INTERVAL_STEPS = int((1.0 / PHYSICS_TIME_STEP) / TARGET_LOG_HZ)

# [추가] 난수 시드 고정
GLOBAL_SEED = 42

def set_global_seed(seed: int = 42):
    """
    Python random / NumPy 난수 시드를 고정해서
    매 실행마다 동일한 에피소드 조건(마찰, 속도, 질량 등)이 나오도록 함.
    """
    random.seed(seed)
    np.random.seed(seed)

# 차량 공기 저항 관련 상수
DRAG_COEFF = 0.82         # 항력 계수 (Cd)
FRONTAL_AREA = 0.0532        # 전면 면적 (m^2)

# 시작 위치
START_POS = [-11.0, -10.7, 0.5]
START_YAW_DEG = 176              
START_YAW_RAD = math.radians(START_YAW_DEG)

# --- 1. 환경 설정 ---
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

# --- 2. 차량 로드 ---
def load_racecar(pos, yaw):
    quat = p.getQuaternionFromEuler([0, 0, yaw])
    car = p.loadURDF("f10_racecar/racecar_differential.urdf", pos, quat)
    
    for _ in range(20):
        p.stepSimulation()
        
    # 물리 제약조건
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
    target_speed = random.uniform(60, 80)
    trigger_dist = random.uniform(0.01, 0.5)
    mass = random.uniform(2.7, 4.34)
    brake_torque = 0.4
    
    return {
        "friction": target_friction,
        "target_speed": target_speed,
        "trigger_dist": trigger_dist,
        "mass": mass,
        "brake_torque": brake_torque
    }

def get_air_density(friction):
    if friction >= 0.9:      return 1.225
    elif friction >= 0.4:    return 1.150
    else:                    return 1.350

# --- 4. 공기 저항력 함수 ---
def apply_drag_force(car_id, air_density):
    vel_vec, _ = p.getBaseVelocity(car_id)
    speed = np.linalg.norm(vel_vec)
    
    if speed < 0.01: return 0.0

    drag_magnitude = 0.5 * air_density * DRAG_COEFF * FRONTAL_AREA * (speed ** 2)
    
    drag_force_vec = [
        -drag_magnitude * (vel_vec[0] / speed),
        -drag_magnitude * (vel_vec[1] / speed),
        -drag_magnitude * (vel_vec[2] / speed)
    ]
    
    p.applyExternalForce(car_id, -1, drag_force_vec, [0, 0, 0], p.LINK_FRAME)
    
    return drag_magnitude

# --- 5. 메인 실행 ---
if __name__ == "__main__":
    # [설정] 난수 시드 고정
    set_global_seed(GLOBAL_SEED)
    print(f"[INFO] Global random seed fixed to {GLOBAL_SEED}")
    print(f"[INFO] Data collection frequency set to {TARGET_LOG_HZ}Hz (Log every {LOG_INTERVAL_STEPS} steps)")

    # [수정] GUI 비활성화 (DIRECT 모드)
    p.connect(p.DIRECT)
    
    plane_id, track_ids, wall_id = setup_environment()
    wall_pos_abs, _ = p.getBasePositionAndOrientation(wall_id)
    wall_x = wall_pos_abs[0]
    
    car_id = load_racecar(START_POS, START_YAW_RAD)
    
    summary_rows = []
    all_step_rows = []
    
    # --- 테이블 헤더 출력 ---
    print(f"\n--- 데이터 수집 시작 (총 {NUM_EPISODES}회, GUI 없음) ---")
    print(f"{'EP':<5} | {'Mass(kg)':<8} | {'Vel(cmd)':<8} | {'Fric(μ)':<7} | {'TrigDist(m)':<11} | {'MaxDrag(N)':<10} | {'Result':<10} | {'FinalDist(m)':<10}")
    print("-" * 105)
    
    wheels = [8, 15]   
    steering = [0, 2] 

    for ep in range(NUM_EPISODES):
        # 초기화
        p.resetBasePositionAndOrientation(car_id, START_POS, p.getQuaternionFromEuler([0, 0, START_YAW_RAD]))
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
            p.setJointMotorControl2(car_id, w, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            
        for _ in range(10):
            p.stepSimulation()
            
        # --- [진행 상황 출력] ---
        print(f"[{ep+1:03d}/{NUM_EPISODES}] | {cond['mass']:<8.1f} | {cond['target_speed']:<8.1f} | {cond['friction']:<7.1f} | {cond['trigger_dist']:<11.2f} | ", end="", flush=True)
        
        sim_time = 0.0
        step_counter = 0  # [수정] 스텝 카운터 초기화
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
            car_vel, _ = p.getBaseVelocity(car_id)
            speed = np.linalg.norm(car_vel)
            
            if speed > max_speed_achieved:
                max_speed_achieved = speed
            
            dist_to_wall = abs(car_pos[0] - wall_x)
            
            # 공기 저항
            drag_force_mag = apply_drag_force(car_id, current_air_density)
            if drag_force_mag > max_drag_val:
                max_drag_val = drag_force_mag
            
            # 트리거
            if dist_to_wall <= cond['trigger_dist']:
                if not is_braking_active:
                    speed_at_trigger = speed
                    time_at_trigger = sim_time
                    
                    ground_ids = [plane_id] + list(track_ids)
                    for gid in ground_ids:
                        p.changeDynamics(gid, -1, lateralFriction=cond['friction'])
                is_braking_active = True
            
            # 조향
            if dist_to_wall <= 1.5: 
                steer_cmd = -1.0
            else: 
                steer_cmd = 0.0

            # 제동/구동
            if is_braking_active:
                brake_torque_cmd = cond['brake_torque']
                for w in wheels:
                    p.setJointMotorControl2(car_id, w, controlMode=p.TORQUE_CONTROL, force=-brake_torque_cmd)
            else:
                brake_torque_cmd = 0.0
                for w in wheels:
                    p.setJointMotorControl2(car_id, w, controlMode=p.VELOCITY_CONTROL, targetVelocity=cond['target_speed'], force=200)
            
            for s in steering:
                p.setJointMotorControl2(car_id, s, controlMode=p.POSITION_CONTROL, targetPosition=steer_cmd)
            
            # [수정] 데이터 저장을 10Hz로 제한 (24스텝마다 저장)
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
            sim_time += PHYSICS_TIME_STEP
            step_counter += 1 # [수정] 스텝 카운트 증가
            
            # 이탈 확인
            if car_pos[2] < -1.0 or car_pos[2] > 2.0:
                is_failure = -1
                break

            # 충돌 확인
            is_collision = False
            if len(p.getContactPoints(car_id, wall_id)) > 0: 
                is_collision = True
            for t_id in track_ids:
                for c in p.getContactPoints(car_id, t_id):
                    if abs(c[7][2]) < 0.7:
                        is_collision = True
                        break
                if is_collision: 
                    break
            
            if is_collision:
                is_failure = 1
                break
            
            # 정지 성공
            if is_braking_active and speed < 0.05:
                if max_speed_achieved < 0.5:
                    is_failure = -1 # 출발 실패
                    break
                is_failure = 0
                stop_distance = dist_to_wall
                break
            
            if sim_time > MAX_SIM_TIME:
                is_failure = -1 # 시간 초과
                break
        
        # 결과 출력
        if is_failure == 0: result_str = "Success"
        elif is_failure == 1: result_str = "Crash"
        else: result_str = "Error"
        
        print(f"{max_drag_val:<10.2f} | {result_str:<10} | {stop_distance:<10.2f}")

        # 데이터 저장
        if is_failure != -1:
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
            summary_rows.append(summary_row)
            for row in episode_steps:
                row["result_is_failure"] = is_failure
            all_step_rows.extend(episode_steps)
            
    p.disconnect()
    
    # --- [수정] Train(80%) / Val(20%) 분할 및 저장 ---
    if summary_rows:
        print("\n--- 데이터 처리 및 분할 저장 중 ---")
        
        df_summary = pd.DataFrame(summary_rows)
        df_steps = pd.DataFrame(all_step_rows)
        
        # 유니크 에피소드 ID 추출
        episodes = df_summary['episode'].unique()
        
        # 무작위로 섞음
        np.random.shuffle(episodes)
        
        # 8:2 지점 계산
        split_idx = int(len(episodes) * 0.8)
        
        train_eps = episodes[:split_idx]
        val_eps = episodes[split_idx:]
        
        print(f"Total Episodes: {len(episodes)}")
        print(f"Train Episodes: {len(train_eps)} (예: {train_eps[:5]} ...)")
        print(f"Val Episodes:   {len(val_eps)} (예: {val_eps[:5]} ...)")
        
        # 데이터프레임 필터링
        train_summary = df_summary[df_summary['episode'].isin(train_eps)]
        val_summary = df_summary[df_summary['episode'].isin(val_eps)]
        
        train_steps = df_steps[df_steps['episode'].isin(train_eps)]
        val_steps = df_steps[df_steps['episode'].isin(val_eps)]
        
        # 파일 저장
        train_summary.to_csv("train_summary.csv", index=False)
        train_steps.to_csv("train_steps.csv", index=False)
        val_summary.to_csv("val_summary.csv", index=False)
        val_steps.to_csv("val_steps.csv", index=False)
        
        print("\n[완료] 4개의 파일이 저장되었습니다:")
        print("  - train_summary.csv, train_steps.csv")
        print("  - val_summary.csv, val_steps.csv")
