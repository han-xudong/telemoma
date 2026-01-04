# read the contents of your README file
from os import path
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
import subprocess
import pathlib

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)


class BuildProto(build_py):
    def run(self):
        proto_dir = pathlib.Path("telemoma/human_interface/proto")
        out_dir = pathlib.Path("telemoma/human_interface")

        for proto in proto_dir.glob("*.proto"):
            subprocess.check_call([
                "python", "-m", "grpc_tools.protoc",
                f"-I{proto_dir}",
                f"--python_out={out_dir}",
                str(proto)
            ])

        super().run()

setup(
    name="telemoma",
    version="0.3.0",
    author="Shivin Dass",
    author_email='shivindass@gmail.com',
    description='A modular and versatile teleoperation system for mobile manipulation',
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/UT-Austin-RobIn/telemoma",
    install_requires=[
        "numpy",
        "opencv-python",
        "mediapipe==0.10.21",
        "scipy",
        "pyrealsense2==2.53.1.4623",
        "pyspacemouse",
        "pynput",
        "pyzmq",
        "matplotlib",
        "gymnasium",
    ],
    packages=find_packages(),
    cmdclass={"build_py": BuildProto},
    python_requires=">=3",
) 
