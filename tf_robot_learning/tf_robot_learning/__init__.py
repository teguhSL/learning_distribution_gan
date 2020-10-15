from . import control
from . import kinematic
from . import distributions
from . import utils
from .utils import param as p
from . import planar_robots
from . import nn
from . import policy

from .utils import tf_utils as tf

datapath = __path__[0][:-17] + 'data'
