import time
import copy
import numpy as np
import pyrealsense2 as rs
from telemoma.human_interface.teleop_core import BaseTeleopInterface, TeleopAction, TeleopObservation
from telemoma.utils.general_utils import run_threaded_command
from telemoma.utils.transformations import quat_diff, quat_to_euler

class T265Reader:
    def __init__(self, sn) -> None:
        self.connection_timeout = 10
        
        self.connection = {
            'right': None,
            'left': None,
        }
        
        self.pipeline = {
            'right': None,
            'left': None,
        }
        
        for side in ['right', 'left']:
            if sn[side] is None:
                continue
            
            start_time = time.time()
            timeout = True
            while self.connection_timeout > time.time() - start_time:
                try:
                    self.pipeline[side] = rs.pipeline()
                    config = rs.config()
                    # config.enable_device(sn[side])
                    config.enable_stream(rs.stream.pose)
                    self.pipeline[side].start(config)
                    self.connection[side] = True
                    timeout = False
                    break
                except:
                    print(f'Connecting to {side} T265....{time.time()-start_time}')
            
            if timeout:
                self.connection[side] = None
                
        for side in ['right', 'left']:
            if self.connection[side] is None:
                print(f'==> Could not connect to {side} T265 with serial number {sn[side]}', 'red')

    def _get_single_pose(self, pipeline):
        # Wait for a coherent pair of frames: depth and pose
        frames = pipeline.wait_for_frames()
        # Get the pose frame and extract translation and rotation
        pose = frames.get_pose_frame().get_pose_data()
        pos = np.array([pose.translation.x, pose.translation.y, pose.translation.z])
        rot = np.array([pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w])
        
        return pos, rot
    
    def get_pose(self):
        
        pose_dict = {
            'right': None,
            'left': None,
        }
        for side in ['right', 'left']:
            if self.connection[side] is None:
                continue
            
            pos, rot = self._get_single_pose(self.pipeline[side])
            pose_dict[side] = dict(pos=pos, quat=rot)
        
        return pose_dict

class T265Policy(BaseTeleopInterface):
    def __init__(
        self,
        sn: dict[str, str] = None,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.position = None
        self.rotation = None
        self.start_input_position = None
        self.start_input_rotation = None
        
        self.position_gain = 1.0
        self.rotation_gain = 1.0
        self.coordinate_change = np.array([[0, 0, -1],
                                           [-1, 0, 0],
                                           [0, 1, 0]])

        self.target_gripper = {'right': 1, 'left': 1}

        self.t265_reader = T265Reader(sn)
        self.reset_state()

    def start(self):
        run_threaded_command(self._update_internal_state)

    def stop(self):
        self.running = False
        
    def reset_state(self) -> None:
        self._state = {
            'right': {
                "pos": None,
                "quat": None,
                "movement_enabled": True if self.t265_reader.connection['right'] is not None else False,
                "controller_on": True,
                "prev_gripper": False,
                "gripper_toggle": False,
                "command": None,
            },

            'left': {
                "pos": None,
                "quat": None,
                "movement_enabled": True if self.t265_reader.connection['left'] is not None else False,
                "controller_on": True,
                "prev_gripper": False,
                "gripper_toggle": False,
                "command": None,
            },
            'buttons': {}
        }
        self.reset_origin = {'right': True, 'left': True}
        self.robot_origin = {'right': None, 'left': None}
        self.t265_origin = {'right': None, 'left': None}
    
    def _update_internal_state(self, num_wait_sec=5):
        last_read_time = time.time()
        while True:
            # Read T265 Pose
            pose = self.t265_reader.get_pose()
            
            for side in pose:
                if pose[side] is None:
                    self._state[side]["controller_on"] = False
                    continue
                else:
                    self._state[side]["controller_on"] = True
                
                pos, quat = pose[side]['pos'], pose[side]['quat']
                
                self._state[side]["pos"] = pos @ self.coordinate_change.T * self.position_gain
                self._state[side]["quat"] = np.r_[quat[:3] @ self.coordinate_change.T, quat[3]]
    
    def _calculate_action(self, robot_obs: dict[str, np.ndarray], side: str) -> np.ndarray:
        t265_state = copy.deepcopy(self._state[side])
        
        delta_action = np.zeros(6)
        print(t265_state)
        if t265_state["movement_enabled"]:
            # Read Observation
            robot_pos = np.array(robot_obs["cartesian_position"][:3])
            robot_quat = robot_obs["cartesian_position"][3:]
            
            # Reset Origin On Release
            if self.reset_origin[side]:
                self.robot_origin[side] = {"pos": robot_pos, "quat": robot_quat}
                self.t265_origin[side] = {"pos": t265_state["pos"], "quat": t265_state["quat"]}
                self.reset_origin[side] = False
                
            if self.robot_origin[side] is None or self.t265_origin[side] is None:
                return None
            
            # Calculate Positional Action
            robot_pos_offset = robot_pos - self.robot_origin[side]["pos"]
            target_pos_offset = t265_state["pos"] - self.t265_origin[side]["pos"]
            pos_action = target_pos_offset - robot_pos_offset
            
            # Calculate Euler Action
            robot_quat_offset = quat_diff(robot_quat, self.robot_origin[side]["quat"])
            target_quat_offset = quat_diff(t265_state["quat"], self.t265_origin[side]["quat"])
            quat_action = quat_diff(target_quat_offset, robot_quat_offset)
            euler_action = quat_to_euler(quat_action)
            
            delta_action = np.concatenate((pos_action, euler_action))
            
        if t265_state["gripper_toggle"]:
            self.target_gripper[side] = 1 - int(robot_obs["gripper_position"] > 0.5)
            
        action = np.concatenate([delta_action, [self.target_gripper[side]]])
        action = action.clip(-1, 1)
        
        return action
        
    def get_action(self, obs: TeleopObservation) -> TeleopAction:
        action = self.get_default_action()
        
        for arm in ['right', 'left']:
            eef_data = obs[arm]
            if eef_data is None:
                continue
            robot_obs = {'cartesian_position': eef_data[:-1], 'gripper_position': eef_data[-1]}
            new_action = self._calculate_action(robot_obs, arm)
            # print(arm, new_action)
            if new_action is not None:
                action[arm] = new_action
        return action
    
    
if __name__ == "__main__":      
    reader = T265Reader({'right': '908412110378', 'left': None}) # replace with your T265 serial numbers
    while 100:
        print(reader.get_pose())
        time.sleep(0.1)