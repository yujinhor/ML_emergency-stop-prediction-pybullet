# === 수정된 전체 코드 ===
import pybullet as p
import pybullet_data
import time
import random
import numpy as np
import pandas as pd

# --- 설정 ---
INITIAL_SPEED_KMH = 40.0
INITIAL_SPEED_MS = INITIAL_SPEED_KMH * 1000 / 3600
ROAD_WIDTH = 3.5
CURVE_RADIUS = 25.0
STRAIGHT_LENGTH = 20.0
CURVE_LENGTH = 20.0
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

# *** 요청에 따른 바퀴 반경 ***
WHEEL_RADIUS = 0.43  # 요청 반영
MAX_DRIVE_TORQUE = 2000.0
MAX_BRAKE_TORQUE = 2500.0

# 디버그 토글
ENABLE_VERBOSE_LOGGING = True
LOG_INTERVAL_SEC = 0.5  # 로그 주기

def setup_environment():
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane_implicit.urdf")
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.4, 0.4, 0.4, 1.0])
    return plane_id

def draw_road_and_track(straight_length, curve_radius, curve_angle_rad, road_width, stop_line_y_offset, brake_points):
    num_dashes = int(straight_length / 2.0)
    for i in range(num_dashes):
        y_start = -straight_length + (i * 2.0)
        y_end = y_start + 1.0
        p.addUserDebugLine([ -road_width/2, y_start, 0.01], [ -road_width/2, y_end, 0.01], [1, 1, 1], 2)
        p.addUserDebugLine([ +road_width/2, y_start, 0.01], [ +road_width/2, y_end, 0.01], [1, 1, 1], 2)
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
    curve_end_y = curve_radius * np.sin(curve_angle_rad) + stop_line_y_offset
    curve_end_x = curve_radius * (1 - np.cos(curve_angle_rad))
    p.addUserDebugLine([curve_end_x - road_width, curve_end_y, 0.01],
                       [curve_end_x + road_width, curve_end_y, 0.01],
                       [1, 0, 0], 5)
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
    print("\n--- '트럭 로봇'(racecar.urdf) 로드 중... ---")
    robot_id = p.loadURDF("racecar/racecar.urdf", basePosition=[0, -STRAIGHT_LENGTH, 0.1])
    num_joints = p.getNumJoints(robot_id)
    joint_name_to_index = {}
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_name_to_index[joint_name] = i
    print(f"--- 로봇 로드 완료. 총 조인트 {num_joints}개 확인. ---")
    steering_joints = [joint_name_to_index['left_steering_hinge_joint'],
                       joint_name_to_index['right_steering_hinge_joint']]
    drive_wheels = [joint_name_to_index['left_rear_wheel_joint'],
                    joint_name_to_index['right_rear_wheel_joint']]
    brake_wheels = [joint_name_to_index['left_rear_wheel_joint'],
                    joint_name_to_index['right_rear_wheel_joint'],
                    joint_name_to_index['left_front_wheel_joint'],
                    joint_name_to_index['right_front_wheel_joint']]
    return robot_id, steering_joints, drive_wheels, brake_wheels

def set_randomized_physics(robot_id, plane_id):
    road_condition = random.choice(["Normal", "Wet", "Icy"])
    friction_map = {"Normal": 0.75, "Wet": 0.45, "Icy": 0.20}
    friction_coefficient = friction_map[road_condition]
    p.changeDynamics(plane_id, -1, lateralFriction=friction_coefficient)
    robot_mass_kg = random.uniform(1900.0, 3100.0)
    p.changeDynamics(robot_id, -1, mass=robot_mass_kg)
    brake_start_y = random.choice(BRAKE_POINTS_Y)
    brake_level = random.uniform(4, 10)
    brake_torque = brake_level * 200.0
    brake_style = random.choice(["Ramp", "Step"])
    wind_force_x = random.uniform(-1000.0, 1000.0)
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
    try:
        base_link_index = -1
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        velocity, ang_vel = p.getBaseVelocity(robot_id)
        rotation_matrix = p.getMatrixFromQuaternion(orn)
        rot_matrix_np = np.array(rotation_matrix).reshape(3, 3)
        inv_rot_matrix_np = rot_matrix_np.T
        local_velocity = inv_rot_matrix_np.dot(velocity)
        speed_forward = local_velocity[1]
        drag_force_magnitude = 0.5 * AIR_DENSITY * TRUCK_FRONTAL_AREA * DRAG_COEFFICIENT * (speed_forward ** 2)
        drag_force_local = [0.0, -np.sign(speed_forward) * drag_force_magnitude, 0.0]
        drag_force_world = rot_matrix_np.dot(drag_force_local)
        p.applyExternalForce(robot_id, base_link_index, drag_force_world.tolist(), pos, p.WORLD_FRAME)
        wind_force_vector = [wind_force_x, 0.0, 0.0]
        p.applyExternalForce(robot_id, base_link_index, wind_force_vector, pos, p.WORLD_FRAME)
    except Exception as e:
        pass

def control_robot(robot_id, current_pos, current_vel, initial_speed, curve_radius, curve_angle_rad,
                  steering_joints, drive_wheels, brake_wheels, brake_torque, brake_start_y, robot_mass_kg,
                  sim_time):
    # Steering PD
    current_y = current_pos[1]
    if current_y <= 0:
        target_x = 0.0
    elif current_y > 0 and current_y < (curve_radius * np.sin(curve_angle_rad)):
        angle = current_y / curve_radius
        target_x = curve_radius * (1 - np.cos(angle))
    else:
        target_x = curve_radius * (1 - np.cos(curve_angle_rad))
    current_x = current_pos[0]
    steering_error = target_x - current_x
    Kp_steering = 0.03
    lateral_velocity = current_vel[0]
    Kd_steering = 0.01
    target_steering_angle = (Kp_steering * steering_error) - (Kd_steering * lateral_velocity)
    target_steering_angle = np.clip(target_steering_angle, -0.4, 0.4)
    for joint in steering_joints:
        p.setJointMotorControl2(robot_id, joint, p.POSITION_CONTROL, targetPosition=target_steering_angle)

    # Path & braking decision
    if current_y <= 0:
        current_path_length = current_y + STRAIGHT_LENGTH
    else:
        angle = np.arcsin(np.clip(current_pos[1] / curve_radius, -1, 1))
        current_path_length = STRAIGHT_LENGTH + (angle * curve_radius)
    brake_start_path_length = brake_start_y + STRAIGHT_LENGTH
    is_braking = current_path_length >= brake_start_path_length

    # Local forward speed
    _, orn = p.getBasePositionAndOrientation(robot_id)
    rotation_matrix = p.getMatrixFromQuaternion(orn)
    rot_matrix_np = np.array(rotation_matrix).reshape(3, 3)
    inv_rot = rot_matrix_np.T
    velocity_world, _ = p.getBaseVelocity(robot_id)
    local_velocity = inv_rot.dot(velocity_world)
    speed_forward = local_velocity[1]

    # speed control (fix: no extra divide by mass)
    speed_error = initial_speed - speed_forward
    k_p_speed = 2.0  # 적절히 낮춰 설정
    desired_accel = k_p_speed * speed_error  # m/s^2

    # drag estimate (signed)
    drag_est = 0.5 * AIR_DENSITY * TRUCK_FRONTAL_AREA * DRAG_COEFFICIENT * (speed_forward ** 2)
    # total required longitudinal force
    total_force = robot_mass_kg * desired_accel + np.sign(speed_forward) * drag_est
    # torque per driven wheel
    torque_per_wheel = (total_force * WHEEL_RADIUS) / max(1, len(drive_wheels))
    torque_per_wheel = np.clip(torque_per_wheel, -MAX_DRIVE_TORQUE, MAX_DRIVE_TORQUE)

    # 반드시 기존 velocity motors 비활성화 (force=0) 후 torque control 적용
    for j in drive_wheels:
        p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, force=0)

    if is_braking:
        applied_brake_torque = -np.clip(brake_torque, 0, MAX_BRAKE_TORQUE)
        for joint in brake_wheels:
            p.setJointMotorControl2(robot_id, joint, p.TORQUE_CONTROL, force=applied_brake_torque)
    else:
        # 주행: torque_per_wheel를 drive_wheels에 부여
        for joint in drive_wheels:
            p.setJointMotorControl2(robot_id, joint, p.TORQUE_CONTROL, force=float(torque_per_wheel))

    # 디버그 로그 (간격 조절은 호출 측에서)
    if ENABLE_VERBOSE_LOGGING and (sim_time % LOG_INTERVAL_SEC) < (PHYSICS_TIME_STEP * 1.1):
        # drag magnitude recalc for printing (positive)
        drag_force_magnitude = 0.5 * AIR_DENSITY * TRUCK_FRONTAL_AREA * DRAG_COEFFICIENT * (speed_forward ** 2)
        wheel_states = p.getJointStates(robot_id, drive_wheels)
        wheel_ang_vels = [s[1] for s in wheel_states] if wheel_states else []
        print(f"[t={sim_time:.2f}] speed_fwd={speed_forward:.3f} m/s, speed_err={speed_error:.3f}, "
              f"torque_wheel={torque_per_wheel:.1f} N·m, brake={'YES' if is_braking else 'NO'}, "
              f"brake_torque={(-applied_brake_torque if is_braking else 0):.1f}, drag={drag_force_magnitude:.1f}, "
              f"wheel_ang_vels={wheel_ang_vels}")

def run_single_simulation(robot_id, plane_id, steering_joints, drive_wheels, brake_wheels):
    initial_cond, brake_torque = set_randomized_physics(robot_id, plane_id)
    start_pos = [0, -STRAIGHT_LENGTH, 0.1]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    p.resetBasePositionAndOrientation(robot_id, start_pos, start_orientation)
    p.resetBaseVelocity(robot_id, linearVelocity=[0, initial_cond["Initial_Speed_ms"], 0], angularVelocity=[0, 0, 0])
    draw_road_and_track(STRAIGHT_LENGTH, CURVE_RADIUS, CURVE_ANGLE_RAD, ROAD_WIDTH, STOP_LINE_Y_OFFSET, BRAKE_POINTS_Y)
    simulation_timeseries_data = []
    step_counter = 0
    is_failure = 0
    is_invalid_run = False
    failure_reason = "N/A"
    robot_mass_kg = initial_cond["Mass_kg"]
    sim_time = 0.0
    max_steps = 240 * 20
    for i in range(max_steps):
        current_pos, current_orn_quat = p.getBasePositionAndOrientation(robot_id)
        current_vel, _ = p.getBaseVelocity(robot_id)
        current_orn_euler = p.getEulerFromQuaternion(current_orn_quat)
        robot_yaw_deg = np.rad2deg(current_orn_euler[2])
        p.resetDebugVisualizerCamera(cameraDistance=7, cameraYaw=robot_yaw_deg, cameraPitch=-15, cameraTargetPosition=current_pos)
        apply_physics(robot_id, initial_cond["Wind_Force_X"])
        control_robot(robot_id, current_pos, current_vel,
                      initial_cond["Initial_Speed_ms"],
                      initial_cond["Curve_Radius_m"], CURVE_ANGLE_RAD,
                      steering_joints, drive_wheels, brake_wheels,
                      initial_cond["Brake_Torque_Nm"], initial_cond["Brake_Start_Y"],
                      robot_mass_kg, sim_time)
        p.stepSimulation()
        time.sleep(1.0 / 60.0)
        step_counter += 1
        sim_time += PHYSICS_TIME_STEP

        if step_counter % DATA_SAMPLING_STEPS == 0:
            wheel_states = p.getJointStates(robot_id, drive_wheels)
            wheel_velocities = [state[1] for state in wheel_states]
            current_snapshot = {
                "timestamp": step_counter * PHYSICS_TIME_STEP,
                "pos_x": current_pos[0],
                "pos_y": current_pos[1],
                "vel_x": current_vel[0],
                "vel_y": current_vel[1],
                "wheel_vel_1": wheel_velocities[0] if len(wheel_velocities) > 0 else 0.0,
                "wheel_vel_2": wheel_velocities[1] if len(wheel_velocities) > 1 else 0.0,
            }
            simulation_timeseries_data.append(current_snapshot)

        if current_vel[1] < -0.1:
            is_invalid_run = True
            failure_reason = "Reversed (Invalid Run)"
            break

        stop_line_y_world = CURVE_RADIUS * np.sin(CURVE_ANGLE_RAD) + STOP_LINE_Y_OFFSET
        if current_pos[1] > (stop_line_y_world + FAILURE_DISTANCE_THRESHOLD):
            is_failure = 1
            failure_reason = "Overshot stop line"
            break

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
            break

        speed = np.linalg.norm(current_vel)
        if speed < 0.1:
            break

    final_pos, _ = p.getBasePositionAndOrientation(robot_id)
    if is_invalid_run:
        print(f"  -> 최종 결과: ** 무효 (Invalid) ** (사유: {failure_reason}) - 이 데이터는 저장되지 않습니다.")
        return None, None
    if is_failure:
        print(f"  -> 최종 결과: ** 실패 (Failure) ** (사유: {failure_reason})")
    else:
        print(f"  -> 최종 결과: 성공 (Success)")
    print(f"     (조건: {initial_cond['Road_Condition']}, 제동시점: {initial_cond['Brake_Start_Y']}m, 제동력: {initial_cond['Brake_Level']:.1f})")
    nn_data_row = list(initial_cond.values()) + [final_pos[0], final_pos[1], is_failure]
    rnn_data_package = {"initial_conditions": initial_cond, "timeseries_data": simulation_timeseries_data, "is_failure": is_failure}
    return nn_data_row, rnn_data_package

if __name__ == "__main__":
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    plane_id = setup_environment()
    robot_id, steering_joints, drive_wheels, brake_wheels = load_robot()
    p.resetDebugVisualizerCamera(cameraDistance=7, cameraYaw=0, cameraPitch=-15, cameraTargetPosition=[0, -STRAIGHT_LENGTH, 0])
    nn_results_data = []
    rnn_all_simulations_data = []
    print("--- 시뮬레이션 시작 ---")
    num_simulations = 20
    for i in range(num_simulations):
        print(f"\n--- 시뮬레이션 {i+1}/{num_simulations} 실행 ---")
        p.removeAllUserDebugItems()
        nn_row, rnn_package = run_single_simulation(robot_id, plane_id, steering_joints, drive_wheels, brake_wheels)
        if nn_row is not None:
            nn_results_data.append(nn_row)
            rnn_all_simulations_data.append(rnn_package)
    print("--- 시뮬레이션 종료 ---")
    p.disconnect()
    if rnn_all_simulations_data:
        nn_columns = list(rnn_all_simulations_data[0]['initial_conditions'].keys()) + ["final_x", "final_y", "is_failure"]
        df = pd.DataFrame(nn_results_data, columns=nn_columns)
        df.to_csv("nn_model_data.csv", index=False)
        print("\nNN 모델용 데이터가 'nn_model_data.csv'에 저장되었습니다.")
        print(df.head())
    else:
        print("시뮬레이션이 실행되지 않아 저장할 데이터가 없습니다. (모든 런이 '무효' 처리되었을 수 있습니다.)")
# === 끝 ===
