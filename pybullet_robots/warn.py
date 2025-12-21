import pybullet as p
import pybullet_data
import time
import math
import random
import numpy as np

# 시뮬레이션 상수
PHYSICS_TIME_STEP = 1.0 / 240.0
NUM_EPISODES = 5000
MAX_SIM_TIME = 100000.0

USE_GUI = True
FOLLOW_CAMERA = True
SLOW_IN_GUI = False

# 난수 설정
GLOBAL_SEED = 42

WAIT_BEFORE_START_SEC = 30.0  
EPISODE_END_WAIT_SEC = 5.0     

#  Runtime Warning 계산 주기
WARN_CHECK_HZ = 10  
          
CAMERA_UPDATE_HZ = 30
COLLISION_CHECK_HZ = 60

WARN_INTERVAL = max(1, int((1.0 / PHYSICS_TIME_STEP) / WARN_CHECK_HZ))
CAM_INTERVAL = max(1, int((1.0 / PHYSICS_TIME_STEP) / CAMERA_UPDATE_HZ))
COLLISION_INTERVAL = max(1, int((1.0 / PHYSICS_TIME_STEP) / COLLISION_CHECK_HZ))

PRINT_ONLY_SUMMARY = True

# 경보 임계값
INITIAL_WARN_THRESH = {
    0.1: 218388,  
    0.5: 43661,
    1.0: 22079
}

RUNTIME_WARN_THRESH = {
    0.1: 0.2238,  
    0.5: 0.0179,
    1.0: 0.0263
}

# 공기 저항 상수
DRAG_COEFF = 0.82
FRONTAL_AREA = 0.0532

START_POS = [-11.0, -10.7, 0.5]
START_YAW_DEG = 176
START_YAW_RAD = math.radians(START_YAW_DEG)

WHEELS = [8, 15]
STEERING = [0, 2]


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def update_follow_camera(car_id):
    car_pos, _ = p.getBasePositionAndOrientation(car_id)
    target_pos = [car_pos[0], car_pos[1], car_pos[2] + 0.5]
    p.resetDebugVisualizerCamera(
        cameraDistance=6.0,
        cameraYaw=50,
        cameraPitch=-25,
        cameraTargetPosition=target_pos
    )


def setup_environment():
    p.setGravity(0, 0, -10)
    p.setTimeStep(PHYSICS_TIME_STEP)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    plane_id = p.loadURDF("plane_implicit.urdf")
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
                           jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0],
                           childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=1, maxForce=10000)

    c = p.createConstraint(car, 10, car, 13, jointType=p.JOINT_GEAR,
                           jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0],
                           childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car, 9, car, 13, jointType=p.JOINT_GEAR,
                           jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0],
                           childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car, 16, car, 18, jointType=p.JOINT_GEAR,
                           jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0],
                           childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=1, maxForce=10000)

    c = p.createConstraint(car, 16, car, 19, jointType=p.JOINT_GEAR,
                           jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0],
                           childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car, 17, car, 19, jointType=p.JOINT_GEAR,
                           jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0],
                           childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car, 1, car, 18, jointType=p.JOINT_GEAR,
                           jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0],
                           childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)

    c = p.createConstraint(car, 3, car, 19, jointType=p.JOINT_GEAR,
                           jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0],
                           childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)

    return car


def get_random_conditions():
    target_friction = random.choice([1.0, 0.5, 0.1])
    target_speed = random.uniform(60, 80)      # init_speed_cmd
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


def compute_initial_metric(mass, init_speed_cmd, friction_cond):
    return (mass * (init_speed_cmd ** 2)) / max(friction_cond, 1e-9)


def compute_runtime_metric(speed, friction_cond, dist_to_wall):
    return (speed / max(friction_cond, 1e-9)) * dist_to_wall


def pick_thresh(table: dict, friction_cond: float):
    keys = list(table.keys())
    k = min(keys, key=lambda x: abs(x - friction_cond))
    return table[k]


SAFE_COLOR = [0.0, 0.85, 0.0, 1.0]
DANGER_COLOR = [0.95, 0.0, 0.0, 1.0]


class BackgroundUI:
    def __init__(self, use_gui: bool, plane_id: int):
        self.use_gui = use_gui
        self.plane_id = plane_id
        self.state = None  # "default" / "safe" / "danger"

        self.default_rgba = [0.4, 0.4, 0.4, 1.0]
        if self.use_gui:
            try:
                vs = p.getVisualShapeData(self.plane_id)
                if vs and len(vs[0]) >= 8:
                    rgba = vs[0][7]
                    if rgba is not None and len(rgba) == 4:
                        self.default_rgba = list(rgba)
            except Exception:
                pass

    def set_default(self):
        if not self.use_gui:
            return
        if self.state == "default":
            return
        p.changeVisualShape(self.plane_id, -1, rgbaColor=self.default_rgba)
        self.state = "default"

    def set_safe(self):
        if not self.use_gui:
            return
        if self.state == "safe":
            return
        p.changeVisualShape(self.plane_id, -1, rgbaColor=SAFE_COLOR)
        self.state = "safe"

    def set_danger(self):
        if not self.use_gui:
            return
        if self.state == "danger":
            return
        p.changeVisualShape(self.plane_id, -1, rgbaColor=DANGER_COLOR)
        self.state = "danger"


def set_vehicle_stopped(car_id):
    for w in WHEELS:
        p.setJointMotorControl2(
            car_id, w,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=0.0,
            force=200
        )
    for s in STEERING:
        p.setJointMotorControl2(
            car_id, s,
            controlMode=p.POSITION_CONTROL,
            targetPosition=0.0
        )


def hold_before_start(car_id, ui: BackgroundUI, initial_warning_on: bool):
    # Initial warning 표시
    if initial_warning_on:
        ui.set_danger()
    else:
        ui.set_safe()

    steps = int(WAIT_BEFORE_START_SEC / PHYSICS_TIME_STEP)
    for i in range(steps):
        set_vehicle_stopped(car_id)

        if USE_GUI and FOLLOW_CAMERA and (i % CAM_INTERVAL == 0):
            update_follow_camera(car_id)

        p.stepSimulation()
        if USE_GUI and SLOW_IN_GUI:
            time.sleep(PHYSICS_TIME_STEP)


def check_collision_fast(car_id, wall_id, track_set):
    contacts = p.getContactPoints(bodyA=car_id)
    for c in contacts:
        bodyB = c[2]
        if bodyB == wall_id:
            return True
        if bodyB in track_set:
            normal_on_b = c[7]
            if abs(normal_on_b[2]) < 0.7:
                return True
    return False


if __name__ == "__main__":
    set_global_seed(GLOBAL_SEED)

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
    else:
        p.connect(p.DIRECT)

    plane_id, track_ids, wall_id = setup_environment()
    ui = BackgroundUI(USE_GUI, plane_id)
    ui.set_default()

    wall_pos_abs, _ = p.getBasePositionAndOrientation(wall_id)
    wall_x = wall_pos_abs[0]

    car_id = load_racecar(START_POS, START_YAW_RAD)
    track_set = set(track_ids)

    for ep in range(NUM_EPISODES):
        p.resetBasePositionAndOrientation(
            car_id, START_POS, p.getQuaternionFromEuler([0, 0, START_YAW_RAD])
        )
        p.resetBaseVelocity(car_id, [0, 0, 0], [0, 0, 0])

        p.changeDynamics(plane_id, -1, lateralFriction=1.0)
        for t_id in track_ids:
            p.changeDynamics(t_id, -1, lateralFriction=1.0)

        # 에피소드 조건
        cond = get_random_conditions()
        friction_cond = cond["friction"]
        init_speed_cmd = cond["target_speed"]
        trigger_dist = cond["trigger_dist"]
        mass = cond["mass"]
        brake_torque = cond["brake_torque"]
        air_density = get_air_density(friction_cond)

        # 차량 세팅
        p.changeDynamics(car_id, -1, mass=mass)
        for i in range(p.getNumJoints(car_id)):
            p.changeDynamics(car_id, i, lateralFriction=1.0)
        p.changeDynamics(car_id, -1, lateralFriction=1.0)

        set_vehicle_stopped(car_id)

        # Initial Warning
        initial_metric = compute_initial_metric(mass, init_speed_cmd, friction_cond)
        initial_thresh = pick_thresh(INITIAL_WARN_THRESH, friction_cond)
        initial_warning_on = (initial_metric >= initial_thresh)
        ENABLE_RUNTIME_WARNING = (not initial_warning_on)
        hold_before_start(car_id, ui, initial_warning_on)

        ui.set_default()

        sim_time = 0.0
        step_counter = 0

        is_braking_active = False
        max_speed_achieved = 0.0

        runtime_warning_active = False 

        last_runtime_metric = None
        last_runtime_thresh = None
        last_runtime_should_warn = None

        is_failure = 0
        stop_distance = 0.0

        while True:
            car_pos, _ = p.getBasePositionAndOrientation(car_id)
            car_vel, _ = p.getBaseVelocity(car_id)

            speed = float(np.linalg.norm(car_vel))
            max_speed_achieved = max(max_speed_achieved, speed)

            dist_to_wall = abs(car_pos[0] - wall_x)

            if USE_GUI and FOLLOW_CAMERA and (step_counter % CAM_INTERVAL == 0):
                update_follow_camera(car_id)

            # 공기 저항 설정
            apply_drag_force(car_id, air_density)

            if ENABLE_RUNTIME_WARNING and (not is_braking_active) and (dist_to_wall > trigger_dist):
                if step_counter % WARN_INTERVAL == 0:
                    runtime_metric = compute_runtime_metric(speed, friction_cond, dist_to_wall)
                    runtime_thresh = pick_thresh(RUNTIME_WARN_THRESH, friction_cond)
                    runtime_should_warn = (runtime_metric <= runtime_thresh)

                    last_runtime_metric = runtime_metric
                    last_runtime_thresh = runtime_thresh
                    last_runtime_should_warn = runtime_should_warn

                    if runtime_should_warn and not runtime_warning_active:
                        runtime_warning_active = True
                        ui.set_danger()
                    elif (not runtime_should_warn) and runtime_warning_active:
                        runtime_warning_active = False
                        ui.set_safe()

            # 정지 트리거
            if (dist_to_wall <= trigger_dist) and (not is_braking_active):
                ground_ids = [plane_id] + list(track_ids)
                for gid in ground_ids:
                    p.changeDynamics(gid, -1, lateralFriction=friction_cond)
                is_braking_active = True

            steer_cmd = -1.0 if dist_to_wall <= 1.5 else 0.0
            for s in STEERING:
                p.setJointMotorControl2(car_id, s, p.POSITION_CONTROL, targetPosition=steer_cmd)

            # 제동 속도 제어, 브레이크 토크 제어
            if is_braking_active:
                for w in WHEELS:
                    w_vel = p.getJointState(car_id, w)[1]
                    if w_vel > 0.5:
                        p.setJointMotorControl2(car_id, w, p.TORQUE_CONTROL, force=-brake_torque)
                    else:
                        p.setJointMotorControl2(car_id, w, p.VELOCITY_CONTROL, targetVelocity=0, force=200)
            else:
                for w in WHEELS:
                    p.setJointMotorControl2(car_id, w, p.VELOCITY_CONTROL, targetVelocity=init_speed_cmd, force=20)

            p.stepSimulation()
            if USE_GUI and SLOW_IN_GUI:
                time.sleep(PHYSICS_TIME_STEP)

            sim_time += PHYSICS_TIME_STEP
            step_counter += 1

            # 이탈 확인
            if car_pos[2] < -1.0 or car_pos[2] > 2.0:
                is_failure = 1
                break

            # 충돌 확인
            if step_counter % COLLISION_INTERVAL == 0:
                if check_collision_fast(car_id, wall_id, track_set):
                    is_failure = 1
                    break

            # 정지 성공
            if is_braking_active and speed < 0.05:
                if max_speed_achieved < 0.5:
                    is_failure = 1
                    break
                is_failure = 0
                stop_distance = dist_to_wall
                break

            if sim_time > MAX_SIM_TIME:
                is_failure = 1
                break

        ui.set_default()

        if PRINT_ONLY_SUMMARY:
            init_state = "DANGER" if initial_warning_on else "SAFE"

            if not ENABLE_RUNTIME_WARNING:
                runtime_part = "Runtime(last)=SKIPPED (initial danger)"
            else:
                if last_runtime_metric is None:
                    runtime_part = "Runtime(last)=N/A (no check before trigger)"
                else:
                    rt_state = "WARN" if last_runtime_should_warn else "SAFE"
                    runtime_part = f"Runtime(last)={rt_state} metric={last_runtime_metric:.4f} thr={last_runtime_thresh}"

            if is_failure == 0:
                result_part = f"RESULT=Success final_dist={stop_distance:.3f}m"
            elif is_failure == 1:
                result_part = "RESULT=Crash"
            else:
                result_part = "RESULT=Error/Aborted"

            print(
                f"[EP {ep+1:04d}] "
                f"Initial={init_state} metric={initial_metric:.2f} thr={initial_thresh} | "
                f"{runtime_part} | "
                f"{result_part}"
            )

        if USE_GUI and EPISODE_END_WAIT_SEC > 0:
            t0 = time.time()
            while (time.time() - t0) < EPISODE_END_WAIT_SEC:
                set_vehicle_stopped(car_id)
                if FOLLOW_CAMERA:
                    update_follow_camera(car_id)
                p.stepSimulation()
                time.sleep(PHYSICS_TIME_STEP)

    p.disconnect()
    print("[DONE] finished")

