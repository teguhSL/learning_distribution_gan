# Generative Adversarial Network to Learn Valid Distributions of Robot Configurations for Inverse Kinematics and  Constrained Motion Planning#

In this work, we use Generative Adversarial Network (GAN) to estimate the distributions of high DoF robot configurations in a constraint manifold. It is then used for speeding up inverse kinematics and sampling-based constrained motion planning. This repository contains the code for this work.

## Installation Procedure ##
Install scipy:
```bash
sudo apt-get install scipy
```

Install tensorflow:
```bash
pip install tensorflow
```


Install networkx:
```bash
pip install networkx
```
Install pinocchio:
```bash
see https://github.com/stack-of-tasks/pinocchio
```

Install transforms3d:
```bash
pip install transforms3d
```

Install pybullet:
```bash
pip install pybullet
```


Then run the following code in the main folder for installing the library :
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
