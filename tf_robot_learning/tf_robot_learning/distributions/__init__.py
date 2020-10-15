from tensorflow_probability import distributions as _distributions

from .poe import PoE
from .mvn import MultivariateNormalFullCovarianceML, MultivariateNormalFullPrecision, MultivariateNormalIso
from .soft_uniform import SoftUniformNormalCdf, SoftUniform
# import from tensorflow_probability
from . import approx


class MultivariateNormalFullCovariance(_distributions.MultivariateNormalFullCovariance):
	def __init__(self,
				 loc=None,
				 covariance_matrix=None,
				 validate_args=False,
				 allow_nan_stats=True,
				 name="MultivariateNormalFullCovariance"):

		self._covariance_matrix = covariance_matrix

		_distributions.MultivariateNormalFullCovariance.__init__(
			self,
			loc,
			covariance_matrix,
			validate_args,
			allow_nan_stats,
			name=name
		)

	@property
	def covariance_matrix(self):
		return self._covariance_matrix

Categorical = _distributions.Categorical
MultivariateNormalDiag = _distributions.MultivariateNormalDiag
MultivariateNormalTriL = _distributions.MultivariateNormalTriL
try:
	Wishart = _distributions.Wishart
except:
	Wishart = None
LogNormal = _distributions.LogNormal
StudentT = _distributions.StudentT
Normal = _distributions.Normal
Uniform = _distributions.Uniform


from .mixture_models import *