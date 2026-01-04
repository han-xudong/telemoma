from telemoma.configs.base_config import teleop_config

teleop_config.arm_right_controller = 't265'
teleop_config.interface_kwargs.t265 = dict(sn = {'right': '908412110378', 'left': None})