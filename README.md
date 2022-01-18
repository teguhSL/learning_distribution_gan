# Learning Constrained Distributions of Robot Configurations with Generative Adversarial Network#

In this work, we use Generative Adversarial Network (GAN) to estimate the distributions of high DoF robot configurations in a constraint manifold. It is then used for speeding up inverse kinematics and sampling-based constrained motion planning. This repository contains the code for this work.

## Installation Procedure ##
Install dependencies:
```bash
pip install tensorflow networkx transforms3d pybullet scipy
```

Install pinocchio:
```bash
see https://github.com/stack-of-tasks/pinocchio
```

Then run the following code in the main folder (tf_robot_learning) for installing the library :
```bash
pip install -e .
```

## How to use the library ##
The library contains general tools for working with probability distributions of robotic systems. For running the specific experiments in the paper, you can look at the following notebooks:
```bash
talos_footfixed.ipynb,
talos_footmoved.ipynb,
2Drobot.ipynb,
panda.ipynb
```
in the notebook folder (tf_robot_learning/notebooks/motion_planning_sampling/).
