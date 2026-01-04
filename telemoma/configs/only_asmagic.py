from telemoma.configs.base_config import teleop_config

teleop_config.arm_right_controller = 'asmagic'
teleop_config.interface_kwargs.asmagic = dict(host={'left':'192.168.31.10', 'right':'192.168.31.11'}, port={'left':8000, 'right':8000})