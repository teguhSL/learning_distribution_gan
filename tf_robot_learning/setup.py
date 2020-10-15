from setuptools import setup, find_packages
import sys


# Setup for Python3
setup(name='tf_robot_learning',
	  version='0.1',
	  description='Tensorflow robotic toolbox',
	  url='',
	  license='MIT',
	  packages=find_packages(),
	  install_requires = ['numpy', 'matplotlib','jupyter', 'tensorflow', 'tensorflow_probability', 'pyyaml', 'lxml'],
	  zip_safe=False)
