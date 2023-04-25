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

This is a bug that needs to be resolved, but everytime you run a robot script below, you will need to stop and re-run this run_server.py program. If you don't, you'll likely get the following error:

```bash
File "/iris/u/rehaan/robostack/r2d2/robot_env.py", line 91, in get_images
return camera_feed[0]['array']
IndexError: list index out of range
```

### NUC (or whichever machine you have an xbox controller connected to)
After activating your conda environment, try collecting a trajectory:

```python
python collect_trajectory.py
```
This will let you control the Franka robot with an Xbox controller and collect a demonstration. 

### Server Machine
Once sufficient demonstrations are collected, you can run franka/bc_franka_img.py. Ensure that the demo is loaded into the env_loader.py under the "bcfrankatest" if condition. Also make sure to go to cfgs/bc_franka_img.yaml and change the mode to either train or eval depending on what you want to do. When in the train mode, the model will save a bunch of snapshots every 1000 steps, during eval load the according one by specifying that in the cfg. 