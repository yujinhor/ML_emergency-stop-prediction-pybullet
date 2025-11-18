import pybullet as p

import pybullet_data

import time

import random

import numpy as np

import pandas as pd



# --- 0. 시뮬레이션 상수 및 변수 정의 ---



# 고정 변수 (PDF 및 사용자 요청 기반)

INITIAL_SPEED_KMH = 40.0  # 초기 속도 (40 km/h)

INITIAL_SPEED_MS = INITIAL_SPEED_KMH * 1000 / 3600  # (m/s)

ROAD_WIDTH = 3.5  # 차로 폭 (m)

CURVE_RADIUS = 25.0  # 도로 곡률 (m)



# 40m 경로 재설계 (직선 20m + 곡선 20m)

STRAIGHT_LENGTH = 20.0 # 직선 주로 길이 (Y = -20 ~ 0)

CURVE_LENGTH = 20.0 # 곡선 주로 길이 (Y = 0 ~ 20)

TOTAL_PATH_LENGTH = STRAIGHT_LENGTH + CURVE_LENGTH # 40m

CURVE_ANGLE_RAD = CURVE_LENGTH / CURVE_RADIUS # 커브 각도 (약 45.8도)



STOP_LINE_Y_OFFSET = 0.0 # 정지선 위치 (커브 끝 = 0)

FAILURE_DISTANCE_THRESHOLD = 0.3 # 실패 기준 (0.3m)

BRAKE_POINTS_Y = [-5.0, 0, 5.0, 10.0] # 급제동 시점 후보 (Y=0이 커브 시작)



# 물리 상수 (사용자 요청)

AIR_DENSITY = 1.225  # 공기 밀도 (kg/m^3)

TRUCK_FRONTAL_AREA = 4.5 # 트럭 전면적 (m^2) (가정: 1.5m x 3.0m)

DRAG_COEFFICIENT = 0.38 # 공기저항계수 (사용자 요청)



# 시뮬레이션 설정

PHYSICS_TIME_STEP = 1.0 / 240.0 # PyBullet 기본값

DATA_SAMPLING_RATE_HZ = 10.0 # 10Hz (0.1초 간격)

DATA_SAMPLING_STEPS = int((1.0 / DATA_SAMPLING_RATE_HZ) / PHYSICS_TIME_STEP) # (1/10) / (1/240) = 24 스텝



# --- 1. PyBullet 환경 설정 함수 ---



def setup_environment():

    """PyBullet 시뮬레이션을 초기화하고 중력 및 '회색 도로' 평면을 설정합니다."""

    p.setGravity(0, 0, -9.81)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

   

    plane_id = p.loadURDF("plane_implicit.urdf")

    p.changeVisualShape(plane_id, -1, rgbaColor=[0.4, 0.4, 0.4, 1.0])

   

    return plane_id



# --- 🚀 [수정된 부분] 🚀 ---

def draw_road_and_track(straight_length, curve_radius, curve_angle_rad, road_width, stop_line_y_offset, brake_points):

    """차선(흰색 점선), 정지선(빨간 실선), 브레이크 마커(초록 실선)를 그립니다."""

   

    # 1. 직선 주로 차선 (Y = -20 ~ 0)

    num_dashes = int(straight_length / 2.0) # 2m 간격 (1m 점선 + 1m 공백)

    for i in range(num_dashes):

        y_start = -straight_length + (i * 2.0)

        y_end = y_start + 1.0 # 1m 길이의 점선

        p.addUserDebugLine([ -road_width/2, y_start, 0.01], [ -road_width/2, y_end, 0.01], [1, 1, 1], 2)

        p.addUserDebugLine([ +road_width/2, y_start, 0.01], [ +road_width/2, y_end, 0.01], [1, 1, 1], 2)



    # 2. 커브길 차선 (Y > 0)

    num_dashes_curve = int(CURVE_LENGTH / 2.0) # 2m 간격 (1m 점선 + 1m 공백)

    for i in range(num_dashes_curve):

        l_start = i * 2.0 # 1m 점선의 시작 (커브 위 0m, 2m, 4m...)

        l_end = l_start + 1.0 # 1m 점선의 끝 (커브 위 1m, 3m, 5m...)

       

        angle1 = l_start / curve_radius

        angle2 = l_end / curve_radius

       

        # 바깥쪽 차선 (R + 3.5/2)

        from_outer_x = (curve_radius + road_width/2) * (1 - np.cos(angle1))

        from_outer_y = (curve_radius + road_width/2) * np.sin(angle1)

        to_outer_x = (curve_radius + road_width/2) * (1 - np.cos(angle2))

        to_outer_y = (curve_radius + road_width/2) * np.sin(angle2)

        p.addUserDebugLine([from_outer_x, from_outer_y, 0.01], [to_outer_x, to_outer_y, 0.01], [1, 1, 1], 2)



        # 안쪽 차선 (R - 3.5/2)

        from_inner_x = (curve_radius - road_width/2) * (1 - np.cos(angle1))

        from_inner_y = (curve_radius - road_width/2) * np.sin(angle1)

        to_inner_x = (curve_radius - road_width/2) * (1 - np.cos(angle2))

        to_inner_y = (curve_radius - road_width/2) * np.sin(angle2)

        p.addUserDebugLine([from_inner_x, from_inner_y, 0.01], [to_inner_x, to_inner_y, 0.01], [1, 1, 1], 2)



    # 3. 정지선 (커브 끝에)

    curve_end_y = curve_radius * np.sin(curve_angle_rad) + stop_line_y_offset

    curve_end_x = curve_radius * (1 - np.cos(curve_angle_rad))

    p.addUserDebugLine([curve_end_x - road_width, curve_end_y, 0.01],

                       [curve_end_x + road_width, curve_end_y, 0.01],

                       [1, 0, 0], 5) # 빨간 실선



    # 4. 브레이크 시점 마커 (초록 실선)

    for brake_dist in brake_points: # brake_dist는 -20, -10, 0, 10

        # 'brake_start_y'가 아닌 '경로상 거리(path length)'로 변환

        path_pos = brake_dist + STRAIGHT_LENGTH # 0m, 10m, 20m, 30m

       

        if path_pos <= STRAIGHT_LENGTH: # 직선 주로 (0, 10, 20m)

            y_pos = path_pos - STRAIGHT_LENGTH

            p.addUserDebugLine([-road_width/2, y_pos, 0.02], [road_width/2, y_pos, 0.02], [0, 1, 0], 2)

        else: # 커브 내부 (30m)

            l_marker = path_pos - STRAIGHT_LENGTH # 10m

            angle_marker = l_marker / curve_radius

           

            # 마커의 중심점 (레일 중앙)

            x_center = curve_radius * (1 - np.cos(angle_marker))

            y_center = curve_radius * np.sin(angle_marker)

           

            # 마커의 방향 (커브의 법선 벡터)

            normal_vec = np.array([-np.cos(angle_marker), np.sin(angle_marker), 0])

           

            # 마커의 양 끝점 계산

            p_inner = np.array([x_center, y_center, 0.02]) + normal_vec * (road_width / 2)

            p_outer = np.array([x_center, y_center, 0.02]) - normal_vec * (road_width / 2)

           

            p.addUserDebugLine(list(p_inner), list(p_outer), [0, 1, 0], 2)

# --- 🚀 [수정 끝] 🚀 ---





def load_robot():

    """

    '트럭 로봇'의 물리적 대리인(racecar.urdf)을 로드하고, 올바른 조인트 인덱스를 찾습니다.

    """

    print("\n--- '트럭 로봇'(물리적 대리인: racecar.urdf) 로드 중... ---")

    try:

        robot_id = p.loadURDF("racecar/racecar.urdf", basePosition=[0, -STRAIGHT_LENGTH, 0.1])

    except Exception as e:

        print(f"!!! 에러: 'racecar/racecar.urdf' 모델을 찾을 수 없습니다: {e}")

        raise



    num_joints = p.getNumJoints(robot_id)

    joint_name_to_index = {}

    for i in range(num_joints):

        joint_info = p.getJointInfo(robot_id, i)

        joint_name = joint_info[1].decode('utf-8')

        joint_name_to_index[joint_name] = i

       

    print(f"--- 로봇 로드 완료. 총 조인트 {num_joints}개 확인. ---")



    try:

        steering_joints = [joint_name_to_index['left_steering_hinge_joint'],

                           joint_name_to_index['right_steering_hinge_joint']]

        drive_wheels = [joint_name_to_index['left_rear_wheel_joint'],

                        joint_name_to_index['right_rear_wheel_joint']]

        brake_wheels = [joint_name_to_index['left_rear_wheel_joint'],

                        joint_name_to_index['right_rear_wheel_joint'],

                        joint_name_to_index['left_front_wheel_joint'],

                        joint_name_to_index['right_front_wheel_joint']]

       

        print("[성공] 올바른 조인트 인덱스를 찾았습니다. (조향, 구동, 제동)")

        return robot_id, steering_joints, drive_wheels, brake_wheels



    except KeyError as e:

        print(f"!!! 에러: 조인트 이름 매핑 실패. {e} 이름을 찾을 수 없습니다.")

        raise





# --- 2. 물리 및 제어 함수 ---



def set_randomized_physics(robot_id, plane_id):

    """시뮬레이션 변수들을 무작위로 설정합니다."""

   

    # 1. 노면 마찰계수 (3개 카테고리)

    road_condition = random.choice(["Normal", "Wet", "Icy"])

    friction_map = {"Normal": 0.75, "Wet": 0.45, "Icy": 0.20}

    friction_coefficient = friction_map[road_condition]

    p.changeDynamics(plane_id, -1, lateralFriction=friction_coefficient)



    # 2. 로봇 질량 (1.9 ~ 3.1톤)

    robot_mass_kg = random.uniform(1900.0, 3100.0)

    p.changeDynamics(robot_id, -1, mass=robot_mass_kg)



    # 3. 급제동 시점 (커브 시작점 Y=0 기준)

    brake_start_y = random.choice(BRAKE_POINTS_Y)



    # 4. 브레이크 레벨 (0~10) -> 토크 (0~10,00 Nm)

    brake_level = random.uniform(4, 10) # 브레이크 최소 보장

    brake_torque = brake_level * 10  # 0~1,000 Nm로 변환

   

    brake_style = random.choice(["Ramp", "Step"])



    # 5. 바람 세기 (X축, 즉 측면)

    wind_force_x = random.uniform(-1000.0, 1000.0) # 최대 1000N의 측풍



    # (고정 변수)

    initial_velocity = INITIAL_SPEED_MS

    curve_radius = CURVE_RADIUS



    initial_conditions = {

        "Road_Condition": road_condition,

        "Friction": friction_coefficient,

        "Mass_kg": robot_mass_kg,

        "Brake_Start_Y": brake_start_y,

        "Brake_Level": brake_level,

        "Brake_Torque_Nm": brake_torque,

        "Brake_Style": brake_style,

        "Wind_Force_X": wind_force_x,

        "Initial_Speed_ms": initial_velocity,

        "Curve_Radius_m": curve_radius

    }

    return initial_conditions, brake_torque



def apply_physics(robot_id, wind_force_x):

    """공기 저항과 바람을 매 스텝 적용합니다."""

    try:

        base_link_index = -1

        pos, orn = p.getBasePositionAndOrientation(robot_id)

        velocity, ang_vel = p.getBaseVelocity(robot_id)

       

        rotation_matrix = p.getMatrixFromQuaternion(orn)

        rot_matrix_np = np.array(rotation_matrix).reshape(3, 3)

        inv_rot_matrix_np = rot_matrix_np.T

        local_velocity = inv_rot_matrix_np.dot(velocity)

        speed_forward = local_velocity[0]



        # 1. 공기 저항 (속도 제곱 비례)

        drag_force_magnitude = 0.5 * AIR_DENSITY * TRUCK_FRONTAL_AREA * DRAG_COEFFICIENT * (speed_forward ** 2)

        drag_force_local = [-drag_force_magnitude, 0, 0]

        drag_force_world = rot_matrix_np.dot(drag_force_local)

        p.applyExternalForce(robot_id, base_link_index, drag_force_world, pos, p.WORLD_FRAME)



        # 2. 바람 (측풍)

        wind_force_vector = [wind_force_x, 0, 0]

        p.applyExternalForce(robot_id, base_link_index, wind_force_vector, pos, p.WORLD_FRAME)

   

    except Exception as e:

        pass



# --- 🚀 [수정된 부분] 🚀 ---

def control_robot(robot_id, current_pos, current_vel, initial_speed, curve_radius, curve_angle_rad, steering_joints, drive_wheels, brake_wheels, brake_torque, brake_start_y):

    """로봇의 조향(PD제어), 구동, 제동을 제어합니다."""

   

    current_y = current_pos[1]

   

    # 1. 조향 제어 (PD Controller: Proportional-Derivative)

    #    '빙글빙글' 도는 현상을 막기 위해 게인 값을 낮춰 안정화

   

    target_x = 0

    # 직선 주로

    if current_y <= 0:

        target_x = 0

    # 커브 구간

    elif current_y > 0 and current_y < (curve_radius * np.sin(curve_angle_rad)):

        angle = current_y / curve_radius # (단순화된 근사)

        target_x = curve_radius * (1 - np.cos(angle))

    # 커브 끝

    else:

        target_x = curve_radius * (1 - np.cos(curve_angle_rad))

           

    current_x = current_pos[0]

   

    # [P term] (Proportional): 중앙에서 얼마나 벗어났는가?

    steering_error = target_x - current_x

    Kp_steering = 0.03 # 🚀 비례 게인 (매우 낮춰 안정화)

   

    # [D term] (Derivative): 중앙을 향해 얼마나 빠르게 움직이는가? (횡방향 속도)

    lateral_velocity = current_vel[0] # X축 속도

    Kd_steering = 0.01 # 🚀 미분 게인 (매우 낮춰 안정화)

   

    # PD 제어기: P항으로 중앙으로 당기고, D항으로 흔들림(횡방향 속도)을 억제

    target_steering_angle = (Kp_steering * steering_error) - (Kd_steering * lateral_velocity)

   

    max_steer = 0.4 # 🚀 최대 조향각 (약 23도)

    target_steering_angle = np.clip(target_steering_angle, -max_steer, max_steer)

   

    for joint in steering_joints:

        p.setJointMotorControl2(robot_id, joint, p.POSITION_CONTROL, targetPosition=target_steering_angle)

    # --- 🚀 [수정 끝] 🚀 ---





    # 2. 구동 및 제동 제어 (기존 로직 유지)

    # 🚀 [수정] 제동 시점을 Y좌표가 아닌, "경로상 거리(path length)"로 계산

    current_path_length = 0

    if current_y <= 0:

        current_path_length = current_y + STRAIGHT_LENGTH # Y=-20 -> 0m, Y=0 -> 20m

    else:

        # Y좌표로 각도 역산 (단순화된 근사)

        angle = np.arcsin(np.clip(current_pos[1] / curve_radius, -1, 1))

        current_path_length = STRAIGHT_LENGTH + (angle * curve_radius) # 직선거리 + 호의 길이

       

    # 'brake_start_y'는 Y좌표가 아니라 "경로 상의 거리"를 의미

    # Y=0 -> 20m, Y=-10 -> 10m, Y=-20 -> 0m, Y=10 -> 30m

    brake_start_path_length = brake_start_y + STRAIGHT_LENGTH

   

    is_braking = current_path_length >= brake_start_path_length

   

    if is_braking:

        # 제동: 4바퀴 모두에 제동 토크 적용

        for joint in drive_wheels:

            p.setJointMotorControl2(robot_id, joint, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        for joint in brake_wheels:

            p.setJointMotorControl2(robot_id, joint, p.TORQUE_CONTROL, force=-brake_torque)

    else:

        # 주행: 초기 속도를 유지하도록 후륜에 토크 적용

        for joint in drive_wheels:

            p.setJointMotorControl2(robot_id, joint, p.VELOCITY_CONTROL, targetVelocity=initial_speed * 1.5, force=1000)





# --- 3. 메인 시뮬레이션 실행 ---



def run_single_simulation(robot_id, plane_id, steering_joints, drive_wheels, brake_wheels):

    """

    한 번의 전체 시뮬레이션을 실행하고, NN용/RNN용 데이터를 반환합니다.

    """

   

    # [A] 시뮬레이션 설정 및 변수 무작위화

    initial_cond, brake_torque = set_randomized_physics(robot_id, plane_id)

   

    # 로봇 초기 위치 및 속도 설정

    start_pos = [0, -STRAIGHT_LENGTH, 0.1] # Y=-20에서 시작

    start_orientation = p.getQuaternionFromEuler([0, 0, 0])

    p.resetBasePositionAndOrientation(robot_id, start_pos, start_orientation)

    p.resetBaseVelocity(robot_id, linearVelocity=[0, initial_cond["Initial_Speed_ms"], 0], angularVelocity=[0, 0, 0])



    # 커브 및 정지선 그리기

    stop_line_y_world = CURVE_RADIUS * np.sin(CURVE_ANGLE_RAD) + STOP_LINE_Y_OFFSET

    draw_road_and_track(STRAIGHT_LENGTH, CURVE_RADIUS, CURVE_ANGLE_RAD, ROAD_WIDTH, STOP_LINE_Y_OFFSET, BRAKE_POINTS_Y)



    # [B] 시뮬레이션 스텝 실행 루프

    simulation_timeseries_data = [] # RNN/LSTM/GRU 모델용 데이터

    step_counter = 0

    is_failure = 0 # 0: 성공, 1: 실패

    is_invalid_run = False # True: 역주행 (데이터 저장 안 함)

    failure_reason = "N/A" # 실패 사유

   

    for i in range(240 * 20): # 최대 20초 시뮬레이션

        current_pos, current_orn_quat = p.getBasePositionAndOrientation(robot_id)

        current_vel, _ = p.getBaseVelocity(robot_id)

       

        # 카메라가 로봇을 뒤에서 따라가도록(추격 시점) 매 스텝 업데이트

        current_orn_euler = p.getEulerFromQuaternion(current_orn_quat)

        robot_yaw_deg = np.rad2deg(current_orn_euler[2])

       

        p.resetDebugVisualizerCamera(

            cameraDistance=7,      # 카메라 거리 (더 가깝게)

            cameraYaw=robot_yaw_deg, # 🚀 로봇의 현재 YAW 각도를 카메라 YAW에 적용

            cameraPitch=-15,       # 카메라 상하 각도 (살짝 위에서)

            cameraTargetPosition=current_pos # 카메라가 로봇의 현재 위치를 바라봄

        )



        # 물리 엔진 적용

        apply_physics(robot_id, initial_cond["Wind_Force_X"])

       

        # 제어 로직 적용 (current_vel을 전달하도록 수정)

        control_robot(robot_id, current_pos, current_vel,

                      initial_cond["Initial_Speed_ms"],

                      initial_cond["Curve_Radius_m"], CURVE_ANGLE_RAD,

                      steering_joints, drive_wheels, brake_wheels,

                      initial_cond["Brake_Torque_Nm"],

                      initial_cond["Brake_Start_Y"])

       

        p.stepSimulation()

        time.sleep(1.0 / 60.0) # 4배속 느리게 재생

        step_counter += 1



        # --- 🔴 RNN/LSTM/GRU 데이터 수집 지점 🔴 ---

        if step_counter % DATA_SAMPLING_STEPS == 0:

            wheel_states = p.getJointStates(robot_id, drive_wheels)

            wheel_velocities = [state[1] for state in wheel_states]

           

            current_snapshot = {

                "timestamp": step_counter * PHYSICS_TIME_STEP,

                "pos_x": current_pos[0],

                "pos_y": current_pos[1],

                "vel_x": current_vel[0],

                "vel_y": current_vel[1],

                "wheel_vel_1": wheel_velocities[0],

                "wheel_vel_2": wheel_velocities[1],

            }

            simulation_timeseries_data.append(current_snapshot)

        # --- 🔴 데이터 수집 끝 🔴 ---



        # [C] 실시간 실패 및 무효 조건 검사 (조기 종료)

       

        # 1. 무효 (역주행): Y 속도가 음수 (뒤로 감)

        if current_vel[1] < -0.1: # -0.1 (m/s) 임계값

            is_invalid_run = True

            failure_reason = "Reversed (Invalid Run)"

            break



        # 2. 실패 (정지선 초과)

        if current_pos[1] > (stop_line_y_world + FAILURE_DISTANCE_THRESHOLD):

            is_failure = 1

            failure_reason = "Overshot stop line"

            break # 즉시 시뮬레이션 종료

       

        # 3. 실패 (차선 이탈)

        target_x = 0 # 실패 판정용 target_x 계산

        if current_pos[1] <= 0:

            target_x = 0

        elif current_pos[1] > 0 and current_pos[1] < (CURVE_RADIUS * np.sin(CURVE_ANGLE_RAD)):

            angle = current_pos[1] / CURVE_RADIUS

            target_x = CURVE_RADIUS * (1 - np.cos(angle))

        else:

            target_x = CURVE_RADIUS * (1 - np.cos(CURVE_ANGLE_RAD))

           

        if abs(current_pos[0] - target_x) > (ROAD_WIDTH / 2):

            is_failure = 1

            failure_reason = "Lane departure"

            break # 즉시 시뮬레이션 종료



        # 4. 정상 종료 (로봇이 멈춤)

        speed = np.linalg.norm(current_vel)

        if speed < 0.1:

            break

           

    # [D] 시뮬레이션 종료 및 결과 판정

    final_pos, _ = p.getBasePositionAndOrientation(robot_id)

   

    # '무효' 런(None)은 건너뜀

    if is_invalid_run:

        print(f"  -> 최종 결과: ** 무효 (Invalid) ** (사유: {failure_reason}) - 이 데이터는 저장되지 않습니다.")

        return None, None # None을 반환하여 메인 루프에서 저장하지 않도록 함

   

    # (루프가 정상 종료되거나, '실패'로 조기 종료된 경우)

    # is_failure가 0 또는 1로 설정되었음

   

    # 시뮬레이션 1회 종료 시마다 콘솔에 결과 출력

    if is_failure:

        print(f"  -> 최종 결과: ** 실패 (Failure) ** (사유: {failure_reason})")

    else:

        # 루프를 돌았는데 실패 사유가 없고 멈췄으면 '성공'

        print(f"  -> 최종 결과: 성공 (Success)")

    print(f"     (조건: {initial_cond['Road_Condition']}, 제동시점: {initial_cond['Brake_Start_Y']}m, 제동력: {initial_cond['Brake_Level']:.1f})")





    nn_data_row = list(initial_cond.values()) + [final_pos[0], final_pos[1], is_failure]

   

    rnn_data_package = {

        "initial_conditions": initial_cond,

        "timeseries_data": simulation_timeseries_data,

        "is_failure": is_failure

    }

   

    return nn_data_row, rnn_data_package



# --- 4. 메인 실행 ---

if __name__ == "__main__":

   

    p.connect(p.GUI) # 시뮬레이션을 눈으로 보기 위해 GUI 모드로 변경

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # 불필요한 GUI 패널 숨기기

   

    plane_id = setup_environment()

   

    try:

        robot_id, steering_joints, drive_wheels, brake_wheels = load_robot()

    except KeyError as e:

        print(f"--- [오류] 조인트 이름 매핑 실패: {e}. 시뮬레이션을 중단합니다. ---")

        p.disconnect()

        exit()

   

    # 카메라 초기 위치 고정 (루프 안에서 업데이트됨)

    p.resetDebugVisualizerCamera(

        cameraDistance=7, cameraYaw=0, cameraPitch=-15, cameraTargetPosition=[0, -STRAIGHT_LENGTH, 0] # 시작점 Y=-20

    )



    nn_results_data = []

    rnn_all_simulations_data = []



    print("--- 시뮬레이션 시작 ---")

   

    num_simulations = 20

    for i in range(num_simulations):

       

        print(f"\n--- 시뮬레이션 {i+1}/{num_simulations} 실행 ---")

       

        # 이전 시뮬레이션의 디버그 라인 삭제

        p.removeAllUserDebugItems()

       

        nn_row, rnn_package = run_single_simulation(

            robot_id, plane_id, steering_joints, drive_wheels, brake_wheels

        )

       

        # '무효' 런(None)은 건너뛰고, '성공' 또는 '실패' 런만 저장

        if nn_row is not None:

            nn_results_data.append(nn_row)

            rnn_all_simulations_data.append(rnn_package)



    print("--- 시뮬레이션 종료 ---")

    p.disconnect()



    # --- 5. 결과 저장 (NN 모델용 CSV) ---

    if rnn_all_simulations_data:

        nn_columns = list(rnn_all_simulations_data[0]['initial_conditions'].keys()) + ["final_x", "final_y", "is_failure"]

        df = pd.DataFrame(nn_results_data, columns=nn_columns)

       

        df.to_csv("nn_model_data.csv", index=False)

       

        print("\nNN 모델용 데이터가 'nn_model_data.csv'에 저장되었습니다.")

        print(df.head())

    else:

        print("시뮬레이션이 실행되지 않아 저장할 데이터가 없습니다. (모든 런이 '무효' 처리되었을 수 있습니다.)")

