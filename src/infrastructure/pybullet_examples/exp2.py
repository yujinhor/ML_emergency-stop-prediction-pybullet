import pybullet as p
import pybullet_data
import time
import random
import numpy as np
import pandas as pd

# --- 0. 시뮬레이션 상수 및 변수 정의 ---

INITIAL_SPEED_KMH = 40.0
INITIAL_SPEED_MS = INITIAL_SPEED_KMH * 1000 / 3600
ROAD_WIDTH = 3.5
CURVE_RADIUS = 25.0

STRAIGHT_LENGTH = 20.0
CURVE_LENGTH = 20.0
TOTAL_PATH_LENGTH = STRAIGHT_LENGTH + CURVE_LENGTH
CURVE_ANGLE_RAD = CURVE_LENGTH / CURVE_RADIUS

STOP_LINE_Y_OFFSET = 0.0
FAILURE_DISTANCE_THRESHOLD = 0.3
BRAKE_POINTS_Y = [-5.0, 0, 5.0, 10.0] 

AIR_DENSITY = 1.225
TRUCK_FRONTAL_AREA = 4.5
DRAG_COEFFICIENT = 0.38

PHYSICS_TIME_STEP = 1.0 / 240.0
DATA_SAMPLING_RATE_HZ = 10.0
DATA_SAMPLING_STEPS = int((1.0 / DATA_SAMPLING_RATE_HZ) / PHYSICS_TIME_STEP)

# --- 1. PyBullet 환경 설정 함수 ---

def setup_environment():
    """PyBullet 시뮬레이션을 초기화하고 중력 및 '회색 도로' 평면을 설정합니다."""
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    plane_id = p.loadURDF("plane_implicit.urdf")
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.4, 0.4, 0.4, 1.0]) 
    
    return plane_id

def draw_road_and_track(straight_length, curve_radius, curve_angle_rad, road_width, stop_line_y_offset, brake_points):
    """차선(흰색 점선), 정지선(빨간 실선), 브레이크 마커(초록 실선)를 그립니다."""
    
    # 1. 직선 주로 차선 (Y = -20 ~ 0)
    num_dashes = int(straight_length / 2.0)
    for i in range(num_dashes):
        y_start = -straight_length + (i * 2.0)
        y_end = y_start + 1.0
        p.addUserDebugLine([ -road_width/2, y_start, 0.01], [ -road_width/2, y_end, 0.01], [1, 1, 1], 2)
        p.addUserDebugLine([ +road_width/2, y_start, 0.01], [ +road_width/2, y_end, 0.01], [1, 1, 1], 2)

    # 2. 커브길 차선 (Y > 0)
    num_dashes_curve = int(CURVE_LENGTH / 2.0)
    for i in range(num_dashes_curve):
        l_start = i * 2.0
        l_end = l_start + 1.0
        
        angle1 = l_start / curve_radius
        angle2 = l_end / curve_radius
        
        from_outer_x = (curve_radius + road_width/2) * (1 - np.cos(angle1))
        from_outer_y = (curve_radius + road_width/2) * np.sin(angle1)
        to_outer_x = (curve_radius + road_width/2) * (1 - np.cos(angle2))
        to_outer_y = (curve_radius + road_width/2) * np.sin(angle2)
        p.addUserDebugLine([from_outer_x, from_outer_y, 0.01], [to_outer_x, to_outer_y, 0.01], [1, 1, 1], 2)

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
                       [1, 0, 0], 5)

    # 4. 브레이크 시점 마커 (초록 실선)
    for brake_dist in brake_points:
        path_pos = brake_dist + STRAIGHT_LENGTH
        
        if path_pos <= STRAIGHT_LENGTH:
            y_pos = path_pos - STRAIGHT_LENGTH
            p.addUserDebugLine([-road_width/2, y_pos, 0.02], [road_width/2, y_pos, 0.02], [0, 1, 0], 2)
        else:
            l_marker = path_pos - STRAIGHT_LENGTH
            angle_marker = l_marker / curve_radius
            
            x_center = curve_radius * (1 - np.cos(angle_marker))
            y_center = curve_radius * np.sin(angle_marker)
            
            normal_vec = np.array([-np.cos(angle_marker), np.sin(angle_marker), 0])
            
            p_inner = np.array([x_center, y_center, 0.02]) + normal_vec * (road_width / 2)
            p_outer = np.array([x_center, y_center, 0.02]) - normal_vec * (road_width / 2)
            
            p.addUserDebugLine(list(p_inner), list(p_outer), [0, 1, 0], 2)

def load_robot():
    """
    로봇을 로드하고, '좌/우'가 명확히 구분된 조인트 인덱스를 반환합니다.
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
        
        left_drive_wheel = joint_name_to_index['left_rear_wheel_joint']
        right_drive_wheel = joint_name_to_index['right_rear_wheel_joint']
        
        left_brake_wheels = [joint_name_to_index['left_rear_wheel_joint'], 
                             joint_name_to_index['left_front_wheel_joint']]
        right_brake_wheels = [joint_name_to_index['right_rear_wheel_joint'], 
                              joint_name_to_index['right_front_wheel_joint']]
        
        drive_wheels = [left_drive_wheel, right_drive_wheel]
        
        print("[성공] 올바른 조인트 인덱스를 찾았습니다. (조향, 구동, 제동)")
        
        return robot_id, steering_joints, drive_wheels, left_brake_wheels, right_brake_wheels

    except KeyError as e:
        print(f"!!! 에러: 조인트 이름 매핑 실패. {e} 이름을 찾을 수 없습니다.")
        raise


# --- 2. 물리 및 제어 함수 ---

def set_randomized_physics(robot_id, plane_id):
    """시뮬레이션 변수들을 무작위로 설정합니다."""
    
    road_condition = random.choice(["Normal", "Wet", "Icy"])
    friction_map = {"Normal": 0.75, "Wet": 0.45, "Icy": 0.20}
    friction_coefficient = friction_map[road_condition]
    p.changeDynamics(plane_id, -1, lateralFriction=friction_coefficient)

    robot_mass_kg = random.uniform(1900.0, 3100.0)
    p.changeDynamics(robot_id, -1, mass=robot_mass_kg)

    brake_start_y = random.choice(BRAKE_POINTS_Y)

    brake_level = random.uniform(4, 8)
    brake_torque = brake_level * 100
    
    brake_style = random.choice(["Ramp", "Step"]) 
    wind_force_x = 0.0

    initial_conditions = {
        "Road_Condition": road_condition,
        "Friction": friction_coefficient,
        "Mass_kg": robot_mass_kg,
        "Brake_Start_Y": brake_start_y,
        "Brake_Level": brake_level,
        "Brake_Torque_Nm": brake_torque,
        "Brake_Style": brake_style,
        "Wind_Force_X": wind_force_x,
        "Initial_Speed_ms": INITIAL_SPEED_MS,
        "Curve_Radius_m": CURVE_RADIUS
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

        drag_force_magnitude = 0.5 * AIR_DENSITY * TRUCK_FRONTAL_AREA * DRAG_COEFFICIENT * (speed_forward ** 2)
        drag_force_local = [-drag_force_magnitude, 0, 0]
        drag_force_world = rot_matrix_np.dot(drag_force_local)
        p.applyExternalForce(robot_id, base_link_index, drag_force_world, pos, p.WORLD_FRAME)

        wind_force_vector = [wind_force_x, 0, 0]
        p.applyExternalForce(robot_id, base_link_index, wind_force_vector, pos, p.WORLD_FRAME)
    
    except Exception as e:
        pass 

def control_robot(robot_id, current_pos, current_vel, initial_speed, curve_radius, curve_angle_rad, 
                  steering_joints, drive_wheels, 
                  left_brake_wheels, right_brake_wheels, 
                  brake_torque, brake_start_y):
    """로봇의 조향(PD제어), 구동, 제동을 제어합니다."""
    
    current_y = current_pos[1]
    
    # --- 1. 조향 제어 (PD Controller) ---
    Kp_steering = 0.5
    Kd_steering = 0.3
    
    target_x = 0
    if current_y <= 0:
        target_x = 0
    else:
        angle = np.arcsin(np.clip(current_y / curve_radius, -1, 1))
        target_x = curve_radius * (1 - np.cos(angle))
            
    current_x = current_pos[0]
    
    steering_error = target_x - current_x
    lateral_velocity = current_vel[0]
    
    target_steering_angle = (Kp_steering * steering_error) - (Kd_steering * lateral_velocity)
    
    max_steer = 0.5
    target_steering_angle = np.clip(target_steering_angle, -max_steer, max_steer)
    
    for joint in steering_joints:
        p.setJointMotorControl2(robot_id, joint, p.POSITION_CONTROL, targetPosition=target_steering_angle)

    # --- 2. 구동 및 제동 제어 ---
    current_path_length = 0
    if current_y <= 0:
        current_path_length = current_y + STRAIGHT_LENGTH
    else:
        angle = np.arcsin(np.clip(current_pos[1] / curve_radius, -1, 1))
        current_path_length = STRAIGHT_LENGTH + (angle * curve_radius)
        
    brake_start_path_length = brake_start_y + STRAIGHT_LENGTH 
    
    is_braking = current_path_length >= brake_start_path_length
    
    if is_braking:
        # [제동]
        for joint in left_brake_wheels:
            p.setJointMotorControl2(robot_id, joint, p.TORQUE_CONTROL, force=-brake_torque)
        for joint in right_brake_wheels:
            p.setJointMotorControl2(robot_id, joint, p.TORQUE_CONTROL, force=brake_torque)
            
    else:
        # [주행] - 크루즈 컨트롤 (등속도 유지) 로직
        target_speed = initial_speed # (11.11 m/s)
        current_forward_speed = current_vel[1]
        speed_error = target_speed - current_forward_speed
        
        Kp_cruise = 10000.0
        base_force = 8000.0  
        
        total_force = base_force + (Kp_cruise * speed_error)
        total_force = np.clip(total_force, 0, 15000) 
        
        p.setJointMotorControl2(robot_id, drive_wheels[0], p.TORQUE_CONTROL, force=total_force)
        p.setJointMotorControl2(robot_id, drive_wheels[1], p.TORQUE_CONTROL, force=-total_force)


# --- 3. 메인 시뮬레이션 실행 ---

def run_single_simulation(robot_id, plane_id, steering_joints, drive_wheels, 
                          left_brake_wheels, right_brake_wheels):
    """
    한 번의 전체 시뮬레이션을 실행하고, NN용/RNN용 데이터를 반환합니다.
    """
    
    initial_cond, brake_torque = set_randomized_physics(robot_id, plane_id)
    
    start_pos = [0, -STRAIGHT_LENGTH, 0.1]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    p.resetBasePositionAndOrientation(robot_id, start_pos, start_orientation)
    p.resetBaseVelocity(robot_id, linearVelocity=[0, initial_cond["Initial_Speed_ms"], 0], angularVelocity=[0, 0, 0])

    stop_line_y_world = CURVE_RADIUS * np.sin(CURVE_ANGLE_RAD) + STOP_LINE_Y_OFFSET
    draw_road_and_track(STRAIGHT_LENGTH, CURVE_RADIUS, CURVE_ANGLE_RAD, ROAD_WIDTH, STOP_LINE_Y_OFFSET, BRAKE_POINTS_Y)

    simulation_timeseries_data = []
    step_counter = 0
    is_failure = 0
    is_invalid_run = False
    failure_reason = "N/A"
    
    for i in range(240 * 20):
        current_pos, current_orn_quat = p.getBasePositionAndOrientation(robot_id)
        current_vel, _ = p.getBaseVelocity(robot_id)
        
        current_orn_euler = p.getEulerFromQuaternion(current_orn_quat)
        robot_yaw_deg = np.rad2deg(current_orn_euler[2])
        
        p.resetDebugVisualizerCamera(
            cameraDistance=7,
            cameraYaw=robot_yaw_deg,
            cameraPitch=-15,
            cameraTargetPosition=current_pos
        )

        apply_physics(robot_id, initial_cond["Wind_Force_X"])
        
        control_robot(robot_id, current_pos, current_vel,
                      initial_cond["Initial_Speed_ms"], 
                      initial_cond["Curve_Radius_m"], CURVE_ANGLE_RAD,
                      steering_joints, drive_wheels, 
                      left_brake_wheels, right_brake_wheels, 
                      initial_cond["Brake_Torque_Nm"],
                      initial_cond["Brake_Start_Y"])
        
        p.stepSimulation()
        time.sleep(1.0 / 60.0)
        step_counter += 1

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

        # [C] 실시간 실패 및 무효 조건 검사 (조기 종료)
        
        if current_vel[1] < -0.1:
            is_invalid_run = True
            failure_reason = "Reversed (Invalid Run)"
            break

        if current_pos[1] > (stop_line_y_world + FAILURE_DISTANCE_THRESHOLD): 
            is_failure = 1
            failure_reason = "Overshot stop line"
            break
        
        target_x = 0
        if current_pos[1] <= 0:
            target_x = 0
        else:
            angle = np.arcsin(np.clip(current_pos[1] / CURVE_RADIUS, -1, 1))
            target_x = CURVE_RADIUS * (1 - np.cos(angle))
            
        if abs(current_pos[0] - target_x) > (ROAD_WIDTH / 2):
            is_failure = 1
            failure_reason = "Lane departure"
            break

        # 🚀 [오류 수정] 'speed' 변수를 정의한 후 사용합니다.
        speed = np.linalg.norm(current_vel)
        if speed < 0.1:
            break
            
    # [D] 시뮬레이션 종료 및 결과 판정
    final_pos, _ = p.getBasePositionAndOrientation(robot_id)
    
    if is_invalid_run:
        print(f"  -> 최종 결과: ** 무효 (Invalid) ** (사유: {failure_reason}) - 이 데이터는 저장되지 않습니다.")
        return None, None
    
    if is_failure:
        print(f"  -> 최종 결과: ** 실패 (Failure) ** (사유: {failure_reason})")
    
    else:
        distance_to_stop_line = stop_line_y_world - final_pos[1]
        SUCCESS_THRESHOLD_M = 3.0
        
        if distance_to_stop_line <= SUCCESS_THRESHOLD_M and distance_to_stop_line >= 0:
            print(f"  -> 최종 결과: ** 성공 (Success) ** (정지선 앞 {distance_to_stop_line:.2f}m 지점에 정지)")
        else:
            print(f"  -> 최종 결과: ** 무효 (Invalid) ** (사유: 정지선에서 너무 멈, {distance_to_stop_line:.2f}m)")
            return None, None

    print(f"    (조건: {initial_cond['Road_Condition']}, 제동시점: {initial_cond['Brake_Start_Y']}m, 제동력: {initial_cond['Brake_Level']:.1f})")


    nn_data_row = list(initial_cond.values()) + [final_pos[0], final_pos[1], is_failure]
    
    rnn_data_package = {
        "initial_conditions": initial_cond,
        "timeseries_data": simulation_timeseries_data,
        "is_failure": is_failure
    }
    
    return nn_data_row, rnn_data_package

# --- 4. 메인 실행 ---
if __name__ == "__main__":
    
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    plane_id = setup_environment()
    
    try:
        robot_id, steering_joints, drive_wheels, left_brake_wheels, right_brake_wheels = load_robot()
    except KeyError as e:
        print(f"--- [오류] 조인트 이름 매핑 실패: {e}. 시뮬레이션을 중단합니다. ---")
        p.disconnect()
        exit() 
    
    p.resetDebugVisualizerCamera(
        cameraDistance=7, cameraYaw=0, cameraPitch=-15, cameraTargetPosition=[0, -STRAIGHT_LENGTH, 0]
    )

    nn_results_data = []
    rnn_all_simulations_data = [] 

    print("--- 시뮬레이션 시작 ---")
    
    num_simulations = 20
    for i in range(num_simulations):
        
        print(f"\n--- 시뮬레이션 {i+1}/{num_simulations} 실행 ---")
        
        p.removeAllUserDebugItems()
        
        nn_row, rnn_package = run_single_simulation(
            robot_id, plane_id, steering_joints, drive_wheels, 
            left_brake_wheels, right_brake_wheels
        )
        
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