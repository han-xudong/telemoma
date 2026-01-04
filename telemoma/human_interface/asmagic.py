import time
import copy
import numpy as np
import zmq
from telemoma.human_interface.teleop_core import BaseTeleopInterface, TeleopAction, TeleopObservation
from telemoma.human_interface import asmagic_msg_pb2
from telemoma.utils.general_utils import run_threaded_command
from telemoma.utils.transformations import quat_diff, quat_to_euler


class PhoneSubscriber:
    def __init__(
        self,
        host: str,
        port: int,
        hwm: int = 1,
        conflate: bool = True,
        timeout: int = 1000,
    ) -> None:
        # Create a ZMQ context
        self.context = zmq.Context()
        # Create a ZMQ subscriber
        self.subscriber = self.context.socket(zmq.SUB)
        # Set high water mark
        self.subscriber.set_hwm(hwm)
        # Set conflate
        self.subscriber.setsockopt(zmq.CONFLATE, conflate)
        # Connect the address
        self.subscriber.connect(f"tcp://{host}:{port}")
        # Subscribe the topic
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        # Set poller
        self.poller = zmq.Poller()
        self.poller.register(self.subscriber, zmq.POLLIN)
        self.timeout = timeout

    def subscribeMessage(self):
        # Receive the message
        if self.poller.poll(self.timeout):
            # Receive the message
            msg = self.subscriber.recv()

            # Parse the message
            phone = asmagic_msg_pb2.Phone()
            phone.ParseFromString(msg)
        else:
            raise RuntimeError("No message received within the timeout period.")
        return (
            phone.timestamp,
            phone.color_img,
            phone.depth_img,
            phone.depth_width,
            phone.depth_height,
            phone.local_pose,
            phone.global_pose,
        )

    def close(self):
        if hasattr(self, "subscriber") and self.subscriber:
            self.subscriber.close()
        if hasattr(self, "context") and self.context:
            self.context.term()


class asMagicReader:
    def __init__(self, host, port) -> None:
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
            if host[side] is None or port[side] is None:
                continue
            
            start_time = time.time()
            timeout = True
            while self.connection_timeout > time.time() - start_time:
                try:
                    self.subscribers[side] = PhoneSubscriber(host[side], port[side])
                    break
                except:
                    print(f'Connecting to {side} asMagic....{time.time()-start_time}')
            
            if timeout:
                self.connection[side] = None
                
        for side in ['right', 'left']:
            if self.connection[side] is None:
                print(f'==> Could not connect to {side} asMagic with host {host[side]} and port {port[side]}')

    def _get_single_pose(self, subscriber):
        # Wait for a coherent pair of frames: depth and pose
        _, _, _, _, _, _, pose = subscriber.subscribeMessage()
        # Get the pose frame and extract translation and rotation
        pose = np.array(pose, dtype=np.float32)

        return pose[:3], pose[3:]

    def get_pose(self):
        
        pose_dict = {
            'right': None,
            'left': None,
        }
        for side in ['right', 'left']:
            if self.connection[side] is None:
                continue

            pos, rot = self._get_single_pose(self.subscribers[side])
            pose_dict[side] = dict(pos=pos, quat=rot)
        
        return pose_dict

class asMagicPolicy(BaseTeleopInterface):
    def __init__(
        self,
        host: dict[str, str] = None,
        port: dict[str, int] = None,
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

        self.asMagic_reader = asMagicReader(host, port)
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
                "movement_enabled": True if self.asMagic_reader.connection['right'] is not None else False,
                "controller_on": True,
                "prev_gripper": False,
                "gripper_toggle": False,
                "command": None,
            },

            'left': {
                "pos": None,
                "quat": None,
                "movement_enabled": True if self.asMagic_reader.connection['left'] is not None else False,
                "controller_on": True,
                "prev_gripper": False,
                "gripper_toggle": False,
                "command": None,
            },
            'buttons': {}
        }
        self.reset_origin = {'right': True, 'left': True}
        self.robot_origin = {'right': None, 'left': None}
        self.asMagic_origin = {'right': None, 'left': None}
    
    def _update_internal_state(self, num_wait_sec=5):
        last_read_time = time.time()
        while True:
            # Read asMagic Pose
            pose = self.asMagic_reader.get_pose()
            
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
        asMagic_state = copy.deepcopy(self._state[side])
        
        delta_action = np.zeros(6)
        print(asMagic_state)
        if asMagic_state["movement_enabled"]:
            # Read Observation
            robot_pos = np.array(robot_obs["cartesian_position"][:3])
            robot_quat = robot_obs["cartesian_position"][3:]
            
            # Reset Origin On Release
            if self.reset_origin[side]:
                self.robot_origin[side] = {"pos": robot_pos, "quat": robot_quat}
                self.asMagic_origin[side] = {"pos": asMagic_state["pos"], "quat": asMagic_state["quat"]}
                self.reset_origin[side] = False
                
            if self.robot_origin[side] is None or self.asMagic_origin[side] is None:
                return None
            
            # Calculate Positional Action
            robot_pos_offset = robot_pos - self.robot_origin[side]["pos"]
            target_pos_offset = asMagic_state["pos"] - self.asMagic_origin[side]["pos"]
            pos_action = target_pos_offset - robot_pos_offset
            
            # Calculate Euler Action
            robot_quat_offset = quat_diff(robot_quat, self.robot_origin[side]["quat"])
            target_quat_offset = quat_diff(asMagic_state["quat"], self.asMagic_origin[side]["quat"])
            quat_action = quat_diff(target_quat_offset, robot_quat_offset)
            euler_action = quat_to_euler(quat_action)
            
            delta_action = np.concatenate((pos_action, euler_action))
            
        if asMagic_state["gripper_toggle"]:
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
    reader = asMagicReader(
        host={'right':'192.168.31.10', 'left':'192.168.31.11'}, # replace with actual IP addresses 
        port={'right':8000, 'left':8000}
    )
    
    while True:
        print(reader.get_pose())
        time.sleep(0.1)