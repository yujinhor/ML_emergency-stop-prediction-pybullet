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
MAX_SIM_TIME = 200.0  # 너무 긴 무한루프 방지용

# 시작 위치 (X = -11.0)
START_POS = [-11.0, -10.7, 0.5]  # [안정화] 높이 0.5로 수정됨
START_YAW_DEG = 176              # 시작 yaw
START_YAW_RAD = math.radians(START_YAW_DEG)

# --- 1. 환경 설정 ---
def setup_environment():
    p.setGravity(0, 0, -10)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 바닥 plane
    plane_id = p.loadURDF("plane_implicit.urdf")
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.4, 0.4, 0.4, 1.0])
    
    # 트랙 + 벽
    track_objects = p.loadSDF("f10_racecar/meshes/barca_track.sdf", globalScaling=1)
    
    wall_id = track_objects[-1]
    track_ids = track_objects[:-1]

    # 트랙 전체 기본 마찰 1.0으로 세팅
    for t_id in track_ids:
        p.changeDynamics(t_id, -1, lateralFriction=1.0)

    # plane도 기본 마찰 1.0
    p.changeDynamics(plane_id, -1, lateralFriction=1.0)
    
    return plane_id, track_ids, wall_id

# --- 2. 차량 로드 ---
def load_racecar(pos, yaw):
    quat = p.getQuaternionFromEuler([0, 0, yaw])
    car = p.loadURDF("f10_racecar/racecar_differential.urdf", pos, quat)
    
    # [안정화] 생성 후 잠시 대기
    for _ in range(50):
        p.stepSimulation()
        
    # 물리 제약조건 (강한 구동력 전달을 위해 설정 유지)
    c = p.createConstraint(car, 9, car, 11, jointType=p.JOINT_GEAR,
                           jointAxis=[0,1,0],
                           parentFramePosition=[0,0,0],
                           childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=1, maxForce=10000)

    c = p.createConstraint(car, 10, car, 13, jointType=p.JOINT_GEAR,
                           jointAxis=[0,1,0],
                           parentFramePosition=[0,0,0],
                           childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car, 9, car, 13, jointType=p.JOINT_GEAR,
                           jointAxis=[0,1,0],
                           parentFramePosition=[0,0,0],
                           childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car, 16, car, 18, jointType=p.JOINT_GEAR,
                           jointAxis=[0,1,0],
                           parentFramePosition=[0,0,0],
                           childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=1, maxForce=10000)

    c = p.createConstraint(car, 16, car, 19, jointType=p.JOINT_GEAR,
                           jointAxis=[0,1,0],
                           parentFramePosition=[0,0,0],
                           childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car, 17, car, 19, jointType=p.JOINT_GEAR,
                           jointAxis=[0,1,0],
                           parentFramePosition=[0,0,0],
                           childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car, 1, car, 18, jointType=p.JOINT_GEAR,
                           jointAxis=[0,1,0],
                           parentFramePosition=[0,0,0],
                           childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)

    c = p.createConstraint(car, 3, car, 19, jointType=p.JOINT_GEAR,
                           jointAxis=[0,1,0],
                           parentFramePosition=[0,0,0],
                           childFramePosition=[0,0,0])
    p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)
    
    return car

# --- 3. 랜덤 파라미터 (질량 + 구동/브레이크 토크 등) ---
def get_random_conditions():
    # 블랙아이스 구간에서의 노면 마찰계수
    target_friction = random.choice([1.0, 0.5, 0.2])
    
    # (옵션) 목표 속도: 제어에 직접 쓰지 않고 피처로만 쓸 수 있음
    target_speed = random.uniform(80, 110)
    
    # 브레이크 시작 거리 (벽으로부터)
    trigger_dist = random.uniform(0.5, 2.0)
    
    # 차량 질량 (1kg ~ 10kg)
    mass = random.uniform(1.0, 10.0)
    
    # 브레이크 토크 (정지 시 음수 토크로 사용)
    brake_torque = 100.0
    
    # 🔥 구동(엑셀) 토크: TORQUE_CONTROL로 가속 제어
    drive_torque = random.uniform(50.0, 150.0)
    
    return {
        "friction": target_friction,
        "target_speed": target_speed,
        "trigger_dist": trigger_dist,
        "mass": mass,
        "brake_torque": brake_torque,
        "drive_torque": drive_torque,
    }

# --- 4. 메인 실행 ---
if __name__ == "__main__":
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    
    plane_id, track_ids, wall_id = setup_environment()
    
    wall_pos_abs, _ = p.getBasePositionAndOrientation(wall_id)
    wall_x = wall_pos_abs[0]
    
    summary_rows = []      # 에피소드 요약용 (에피소드 단위 피처 + 결과)
    all_step_rows = []     # GRU/Attention용 시계열 (step 단위 로그)
    
    print(f"--- 데이터 수집 시작 (Black Ice + 질량 + 토크 제동 시나리오) ---")
    
    for ep in range(NUM_EPISODES):
        # 이전 차량 제거
        if 'car_id' in locals():
            p.removeBody(car_id)

        car_id = load_racecar(START_POS, START_YAW_RAD)
        cond = get_random_conditions()
        
        # [초기화] 출발 전에는 무조건 마찰력 1.0 (Dry) 적용
        p.changeDynamics(plane_id, -1, lateralFriction=1.0)
        for t_id in track_ids:
            p.changeDynamics(t_id, -1, lateralFriction=1.0)

        # 차량 접촉 마찰도 기본값으로
        for i in range(p.getNumJoints(car_id)):
            p.changeDynamics(car_id, i, lateralFriction=1.0)
        p.changeDynamics(car_id, -1, lateralFriction=1.0)

        # 차량 질량 적용
        p.changeDynamics(car_id, -1, mass=cond['mass'])
        
        print(
            f"[{ep+1}/{NUM_EPISODES}] "
            f"m:{cond['mass']:.1f}kg, v_cmd:{cond['target_speed']:.1f}, "
            f"μ_black:{cond['friction']}, trigger:{cond['trigger_dist']:.2f}m, "
            f"T_drive:{cond['drive_torque']:.1f}, T_brake:{cond['brake_torque']:.1f}",
            end=""
        )
        
        wheels = [8, 15]   # 구동 휠
        steering = [0, 2]  # 조향 휠 (전륜)

        sim_time = 0.0
        is_failure = 0
        stop_distance = 0.0
        max_speed_achieved = 0.0
        
        is_braking_active = False

        # GRU/Attention용 per-step 로그
        episode_steps = []

        # 트리거 시점 정보
        speed_at_trigger = None
        time_at_trigger = None
        dist_at_trigger = None
        time_to_stop = None
        
        while True:
            car_pos, _ = p.getBasePositionAndOrientation(car_id)
            car_vel, _ = p.getBaseVelocity(car_id)
            speed = np.linalg.norm(car_vel)  # 🔥 해당 시점 속도 (이미 수집 중)
            
            if speed > max_speed_achieved:
                max_speed_achieved = speed
            
            dist_to_wall = abs(car_pos[0] - wall_x)
            
            # --- 트리거 & 블랙아이스 로직 ---
            if dist_to_wall <= cond['trigger_dist']:
                if not is_braking_active:
                    # 트리거 시점 기록 (🔥 이 시점의 속도도 요약에 포함)
                    speed_at_trigger = speed
                    time_at_trigger = sim_time
                    dist_at_trigger = dist_to_wall

                    # 이 순간 plane + barca_track 전체 마찰력을 cond['friction']로 낮춤
                    ground_ids = [plane_id] + list(track_ids)
                    for gid in ground_ids:
                        p.changeDynamics(gid, -1, lateralFriction=cond['friction'])
                is_braking_active = True
            
            # --- 벽과의 거리 1.5m 이하에서 전륜 우측 조향 ---
            if dist_to_wall <= 1.5:
                steer_cmd = -1.0  # 라디안 단위, 오른쪽(또는 왼쪽일 수 있음)
            else:
                steer_cmd = 0.0

            # --- 구동 / 제동 제어 (모두 TORQUE_CONTROL) ---
            if is_braking_active:
                # 블랙아이스 이후: 제동 토크 (음수)
                motor_torque_cmd = -cond['brake_torque']
            else:
                # 블랙아이스 전: 구동 토크 (양수)
                motor_torque_cmd = cond['drive_torque']

            # 실제 휠에 토크 적용
            for w in wheels:
                p.setJointMotorControl2(
                    car_id, w,
                    controlMode=p.TORQUE_CONTROL,
                    force=motor_torque_cmd
                )
            
            # 전륜 조향 적용
            for s in steering:
                p.setJointMotorControl2(
                    car_id, s,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=steer_cmd
                )
                
            # 현재 노면 마찰 (트리거 전: 1.0, 후: cond['friction'])
            current_friction = cond['friction'] if is_braking_active else 1.0

            # --- 시계열 로그 쌓기 ---
            # speed(시점별 속도) 이미 포함됨
            step_row = {
                "episode": ep,
                "time": sim_time,
                "x": car_pos[0],
                "y": car_pos[1],
                "z": car_pos[2],
                "speed": speed,                       # 🔥 해당 시점 속도
                "dist_to_wall": dist_to_wall,
                "current_friction": current_friction,
                "is_braking": int(is_braking_active),
                "wheel_torque_cmd": motor_torque_cmd, # 현재 적용된 휠 토크(가속/제동 공통)
                "steer_cmd": steer_cmd,
                "mass_kg": cond['mass'],
                "friction": cond['friction'],
                "init_speed_cmd": cond['target_speed'],
                "trigger_dist_m": cond['trigger_dist'],
                "base_brake_torque": cond['brake_torque'],
                "drive_torque": cond['drive_torque'],
            }
            episode_steps.append(step_row)
            
            p.stepSimulation()
            sim_time += PHYSICS_TIME_STEP
            
            p.resetDebugVisualizerCamera(5.0, 90, -40, car_pos)
            
            # [디버깅] 이탈 확인 (Z축 체크)
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
                    # contactNormal z성분이 작으면 수직벽에 가까움 → 충돌로 처리
                    if abs(c[7][2]) < 0.7:
                        is_collision = True
                        break
                if is_collision:
                    break
            
            if is_collision:
                is_failure = 1
                print(f" -> 💥 충돌! (변환지점: {cond['trigger_dist']:.2f}m)")
                break
            
            # 정지 성공 판정
            if is_braking_active and speed < 0.1:
                if max_speed_achieved < 0.5:
                    print(" -> ⚠️ 출발 실패")
                    is_failure = -1
                    break
                
                is_failure = 0
                stop_distance = dist_to_wall
                if time_at_trigger is not None:
                    time_to_stop = sim_time - time_at_trigger
                else:
                    time_to_stop = None
                print(f" -> ✅ 정지 성공 (최종거리: {stop_distance:.2f}m)")
                break
            
            # 시뮬레이션 시간 제한
            if sim_time > MAX_SIM_TIME:
                print(" -> ⏰ 시간 초과")
                is_failure = -1
                break
        
        # 유효한 에피소드만 저장
        if is_failure != -1:
            summary_row = {
                "episode": ep,
                "mass_kg": cond['mass'],
                "friction": cond['friction'],
                "init_speed_cmd": cond['target_speed'],
                "trigger_dist_m": cond['trigger_dist'],
                "brake_torque": cond['brake_torque'],
                "drive_torque": cond['drive_torque'],
                "result_is_failure": is_failure,
                "final_dist_to_wall": stop_distance if is_failure == 0 else 0.0,
                "max_speed_achieved": max_speed_achieved,
                "speed_at_trigger": speed_at_trigger if speed_at_trigger is not None else 0.0,  # 🔥 트리거 시점 속도
                "time_at_trigger": time_at_trigger if time_at_trigger is not None else 0.0,
                "dist_at_trigger": dist_at_trigger if dist_at_trigger is not None else 0.0,
                "time_to_stop": time_to_stop if time_to_stop is not None else 0.0,
            }
            summary_rows.append(summary_row)

            # step 로그에도 결과 라벨 붙이기
            for row in episode_steps:
                row["result_is_failure"] = is_failure
            all_step_rows.extend(episode_steps)
        else:
            # 출발 실패 / 시간 초과 / 이탈 등은 버림
            pass
            
    p.disconnect()
    
    # --- CSV 저장 ---
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        df_sum.to_csv("black_ice_data.csv", index=False)
        print("\n[완료] 'black_ice_data.csv' 저장됨.")
        print(df_sum.head())
    else:
        print("요약 데이터 없음 (summary).")

    if all_step_rows:
        df_steps = pd.DataFrame(all_step_rows)
        df_steps.to_csv("black_ice_steps.csv", index=False)
        print("\n[완료] 'black_ice_steps.csv' 저장됨.")
        print(df_steps.head())
    else:
        print("시계열 데이터 없음 (steps).")
