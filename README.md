# R2D2 Robot Stack

The repository provides the code for contributing to and using the R2D2 dataset.

NOTE: This repository has two dependencies listed below. If you are setting this up on the robot NUC, (1) is required

(1) https://github.com/facebookresearch/fairo


## Setup Guide
Setup this repository on both the server and client machine (ie: NUC and workstation). The NUC acts as the server and the "client machine" can be another machine on the local network including the NUC itself. 

Use the environment.yml for the environment setup

Regardless of the machine, go into r2d2/misc/parameters.py, and:
- Set robot_ip to match the IP address of your robot
- Set nuc_ip to match the IP address of your NUC

If you are setting this up on the robot NUC:
- In r2d2/misc/parameters.py, set "sudo_password" to your machine's corresponding sudo password. Sudo access is needed to launch the robot. The rest of the parameters can be ignored for now.
## Usage

### Server Machine
Activate the polymetis conda environment:

```bash
conda activate polymetis-local
```

Start the server:

```python
python scripts/server/run_server.py
```

### Client Machine or NUC
After activating your conda environment, try collecting a trajectory:

```python
python collect_trajectory.py
```
This will let you control the Franka robot with an Xbox controller and collect a demonstration. Once sufficient demonstrations are collected, you can run franka/bc_franka_img.py. Ensure that the demo is loaded into the env_loader.py under the "bcfrankatest" if condition. 