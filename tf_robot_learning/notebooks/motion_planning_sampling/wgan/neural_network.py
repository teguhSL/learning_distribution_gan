import tensorflow as tf
from tensorflow_probability import distributions as ds
import numpy as np

# random_rademacher : Generates Tensor consisting of -1 or +1, chosen uniformly at random.
from tensorflow_probability.python.math import random_rademacher

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import ReLU, Dropout
from tensorflow.keras.optimizers import Adam


# create a function which uses Xavier initialization to initialize the weights of the network.
# http://proceedings.mlr.press/v9/glorot10a.html
def xavier(shape, dtype=tf.float64):
    return tf.random.truncated_normal(
        shape,
        mean=0.0,
        stddev=np.sqrt(2/sum(shape)), dtype=dtype)


class NonBayesianDenseLayer(tf.keras.Model):
    """A fully-connected non-Bayesian neural network layer

    Parameters
    ----------
    d_in : int
        Dimensionality of the input (# input features)
    d_out : int
        Dimensionality of the output of the current layer (# layer output dimension )
    name : str
        Name for the layer

    Attributes
    ----------
    losses : tensorflow.Tensor
        Sum of the Kullback–Leibler divergences between
        the posterior distributions and their priors

    Methods
    -------
    call : tensorflow.Tensor
        Perform the forward pass of the data through
        the layer
    """

    def __init__(self, d_in, d_out, name=None, initialize_zeros=False, dtype="float64"):
        super().__init__(name=name, dtype=dtype)
        self.d_in = d_in
        self.d_out = d_out
        alpha = 1.
        if initialize_zeros:
            alpha = 1e-2

        self.w_loc = tf.Variable(xavier([d_in, d_out], dtype=self.dtype)*alpha, name='w_loc')
        self.b_loc = tf.Variable(xavier([1, d_out], dtype=self.dtype)*alpha, name='b_loc')

    def call(self, x):
        """Perform the forward pass"""
        return x @ self.w_loc + self.b_loc


class NonBayesianDenseNetwork(tf.keras.Model):
    """A multilayer fully-connected non-Bayesian neural network

    Parameters
    ----------
    dims : List[int]
        List of units in each layer
    name : str
        Name for the network

    Attributes
    ----------
    losses : tensorflow.Tensor
        Sum of the Kullback–Leibler divergences between
        the posterior distributions and their priors,
        over all layers in the network

    Methods
    -------
    call : tensorflow.Tensor
        Perform the forward pass of the data through
        the network
    """

    def __init__(self, dims, act=tf.nn.relu, final_layer=None, dropout=None, initialize_zeros=False, name=None, dtype=tf.float64):

        super().__init__(name=name, dtype=dtype)
        # Output dimension
        self.d_out = dims[-1]
        self.dropout = dropout
        if final_layer is None:
            final_layer = lambda x: x
        self.steps = []
        self.acts = []
        for i in range(len(dims) - 1):
            self.steps += [NonBayesianDenseLayer(dims[i], dims[i + 1], initialize_zeros=initialize_zeros)]
            if isinstance(act, list):
                self.acts += [act[i]]
            else:
                self.acts += [act]

        self.acts[-1] = final_layer

    def call(self, x):
        """Perform the forward pass"""

        for i in range(len(self.steps)):
            x = self.steps[i](x)
            x = self.acts[i](x)
            if i < len(self.steps) and self.dropout is not None:
                x = tf.nn.dropout(x, rate=self.dropout)

        return x

    def test(self, x):
        """Perform the forward pass without dropout"""

        for i in range(len(self.steps)):
            x = self.steps[i](x)
            x = self.acts[i](x)
        return x



class MLP(tf.keras.Model):

    def __init__(self, dims, batch_size=1, act=tf.nn.relu, final_layer=None, dropout=None, initialize_zeros=False,
                 name=None,  dtype=tf.float64):

        self.batch_size = batch_size
        super().__init__(name=name, dtype=dtype)
        # Multilayer fully-connected neural network to predict mean
        self.loc_net_list = []
        for i in range(self.batch_size):
            self.loc_net_list.append(NonBayesianDenseNetwork(dims, act=act, final_layer=final_layer, dropout=dropout, initialize_zeros=initialize_zeros))

    def forward_pass(self, x):
        return tf.stack([self.loc_net_list[i](x) for i in range(self.batch_size)])

    def forward_pass_test(self, x): # without dropout
        return tf.stack([self.loc_net_list[i].test(x) for i in range(self.batch_size)])

    def forward_pass_batch(self, x):
        return tf.stack([self.loc_net_list[i](x[i]) for i in range(self.batch_size)])

################################################ EQUATION LEARNER MODEL ###########################################


def eq_learner_activation(z, v):
    assert v == z.shape[-1]//6, "not a valid v"
    y0 = z[..., 0:v]
    y1 = tf.math.sin(z[..., v:2*v])
    y2 = tf.math.cos(z[..., 2*v:3*v])
    y3 = tf.keras.activations.sigmoid(z[..., 3*v:4*v])
    y4 = z[..., 4*v:5*v] * z[..., 4*v:5*v]

    return tf.concat([y0, y1, y2, y3, y4], axis=-1)


class EquationLearnerDenseNetwork(tf.keras.Model):
    """A multilayer trigonometric non-Bayesian neural network https://arxiv.org/abs/1610.02995

    Parameters
    ----------
    dims : List[int]
        List of units in each layer
    name : str
        Name for the network

    Attributes
    ----------
    losses : tensorflow.Tensor
        Sum of the Kullback–Leibler divergences between
        the posterior distributions and their priors,
        over all layers in the network

    Methods
    -------
    call : tensorflow.Tensor
        Perform the forward pass of the data through
        the network
    """

    def __init__(self, vs, dims, hidden_layer_size, final_layer=None, dropout=None, initialize_zeros=False, name=None,
                 dtype=tf.float64):
        """

        :param dims:dims correspond to [x_dim] + [u_i+2v_i for each i] + [y_dim] and len(v_list) determines the number of hidden layers
        :param final_layer:
        :param dropout:
        :param initialize_zeros:
        :param name:
        :param dtype:
        """
        super().__init__(name=name, dtype=dtype)
        # Output dimension
        self.hidden_layer_size = hidden_layer_size
        self.vs = vs
        self.op_number = range(hidden_layer_size + 1)
        self.d_out = dims[-1]
        self.dropout = dropout
        self.steps = []
        if final_layer is None:
            self.final_layer_act = lambda x: x
        else:
            self.final_layer_act = final_layer

        for i in self.op_number:
            self.steps += [NonBayesianDenseLayer(dims[2*i], dims[2*i + 1], initialize_zeros=initialize_zeros)]

    def call(self, x):
        """Perform the forward pass"""

        for i in self.op_number:
            x = self.steps[i](x)
            if i == self.hidden_layer_size:
                x = self.final_layer_act(x)
            else:
                x = eq_learner_activation(x, self.vs[i])
            if i < len(self.steps) and self.dropout is not None:
                x = tf.nn.dropout(x, rate=self.dropout)

        return x

    def test(self, x):
        """Perform the forward pass without dropout"""

        for i in self.op_number:
            x = self.steps[i](x)
            if i == self.hidden_layer_size:
                x = self.final_layer_act(x)
            else:
                x = eq_learner_activation(x, self.vs[i])
        return x

class EquationLearnerNetwork(tf.keras.Model):
    """
    We choose always u = 4*v. In each layer, v should be given.
    It corresponds to the number of neurons of type (u=4, v=1) described in the paper
    Activation takes u+2v (6*v) elements and returns u+v(5*v) elements
    dims correspond to [x_dim] + v_list + [y_dim] and len(v_list) determines the number of hidden layers
    """
    def __init__(self, dims, batch_size=1, final_layer=None, dropout=None, initialize_zeros=False,
                 name=None,  dtype=tf.float64):

        vs = dims[1:-1]
        hidden_layer_size = len(vs)

        dims_ = [dims[0]]
        for v in dims[1:-1]:
            dims_ += [6*v] + [5*v]
        dims_ += [dims[-1]]

        self.batch_size = batch_size
        super().__init__(name=name, dtype=dtype)
        # Multilayer fully-connected neural network to predict mean
        self.loc_net_list = []
        for i in range(self.batch_size):

            self.loc_net_list.append(EquationLearnerDenseNetwork(vs, dims_, hidden_layer_size=hidden_layer_size,
                                                                 final_layer=final_layer,
                                                                 dropout=dropout, initialize_zeros=initialize_zeros))

    def forward_pass(self, x):
        return tf.stack([self.loc_net_list[i](x) for i in range(self.batch_size)])

    def forward_pass_test(self, x): # without dropout
        return tf.stack([self.loc_net_list[i].test(x) for i in range(self.batch_size)])

    def forward_pass_batch(self, x):
        return tf.stack([self.loc_net_list[i](x[i]) for i in range(self.batch_size)])

#################################################### BAYESIAN MODEL #################################################
class BayesianDenseLayer(tf.keras.Model):
    """A fully-connected Bayesian neural network layer
    Outputs a Gaussian distribution
    Parameters
    ----------
    d_in : int
        Dimensionality of the input (# input features)
    d_out : int
        Dimensionality of the output of the current layer (# layer output dimension )
    name : str
        Name for the layer

    Attributes
    ----------
    losses : tensorflow.Tensor
        Sum of the Kullback–Leibler divergences between
        the posterior distributions and their priors

    Methods
    -------
    call : tensorflow.Tensor
        Perform the forward pass of the data through
        the layer
    """

    def __init__(self, d_in, d_out, name=None, initialize_zeros=False):

        super(BayesianDenseLayer, self).__init__(name=name)
        self.d_in = d_in
        self.d_out = d_out
        alpha = 1
        if initialize_zeros:
            alpha = 1e-2

        self.w_loc = tf.Variable(xavier([d_in, d_out])*alpha, name='w_loc')
        self.w_std = tf.Variable((xavier([d_in, d_out]) - 6.0)*alpha, name='w_std')
        self.b_loc = tf.Variable(xavier([1, d_out])*alpha, name='b_loc')
        self.b_std = tf.Variable((xavier([1, d_out]) - 6.0)*alpha, name='b_std')

    def call(self, x):
        """Perform the forward pass"""

        # Flipout-estimated weight samples
        s = random_rademacher(tf.shape(x))
        r = random_rademacher([x.shape[0], self.d_out])
        w_samples = tf.nn.softplus(self.w_std) * tf.random.normal([self.d_in, self.d_out])
        w_perturbations = r * tf.matmul(x * s, w_samples)
        w_outputs = tf.matmul(x, self.w_loc) + w_perturbations

        # Flipout-estimated bias samples
        r = random_rademacher([x.shape[0], self.d_out])
        b_samples = tf.nn.softplus(self.b_std) * tf.random.normal([self.d_out])
        b_outputs = self.b_loc + r * b_samples

        return w_outputs + b_outputs


    @property
    def losses(self):
        """Sum of the KL divergences between priors + posteriors"""
        weight = ds.Normal(self.w_loc, tf.nn.softplus(self.w_std))
        bias = ds.Normal(self.b_loc, tf.nn.softplus(self.b_std))
        prior = ds.Normal(0, 1)
        return (tf.reduce_sum(ds.kl_divergence(weight, prior)) +
                tf.reduce_sum(ds.kl_divergence(bias, prior)))


class BayesianDenseNetwork(tf.keras.Model):
    """A multilayer fully-connected Bayesian neural network

    Parameters
    ----------
    dims : List[int]
        List of units in each layer
    name : str
        Name for the network

    Attributes
    ----------
    losses : tensorflow.Tensor
        Sum of the Kullback–Leibler divergences between
        the posterior distributions and their priors,
        over all layers in the network

    Methods
    -------
    call : tensorflow.Tensor
        Perform the forward pass of the data through
        the network
    """

    def __init__(self, dims, act=tf.nn.relu, final_layer=None, dropout=False, initialize_zeros=False, name=None):

        super(BayesianDenseNetwork, self).__init__(name=name)
        # Output dimension
        self.d_out = dims[-1]
        self.dropout = dropout
        if final_layer is None:
            final_layer = lambda x: x
        self.steps = []
        self.acts = []
        for i in range(len(dims) - 1):
            self.steps += [BayesianDenseLayer(dims[i], dims[i + 1])]
            if isinstance(act, list):
                self.acts += [act[i]]
            else:
                self.acts += [act]

        self.acts[-1] = final_layer

    def call(self, x):
        """Perform the forward pass"""

        for i in range(len(self.steps)):
            x = self.steps[i](x)
            x = self.acts[i](x)
            if i < len(self.steps) and self.dropout:
                x = tf.nn.dropout(x, rate=0.3)

    def test(self, x):
        """Perform the forward pass without dropout"""

        for i in range(len(self.steps)):
            x = self.steps[i](x)
            x = self.acts[i](x)
        return x

    @property
    def losses(self):
        """Sum of the KL divergences between priors + posteriors"""
        return tf.reduce_sum([s.losses for s in self.steps])


class BayesianMLP(tf.keras.Model):

    def __init__(self, dims, batch_size=1, act=tf.nn.relu, final_layer=None, dropout=False, initialize_zeros=False, name=None):
        self.batch_size = batch_size
        super().__init__(name=name)
        # Multilayer fully-connected neural network to predict mean
        self.loc_net_list = []
        for i in range(self.batch_size):
            self.loc_net_list.append(BayesianDenseNetwork(dims, act=act, final_layer=final_layer, dropout=dropout, initialize_zeros=initialize_zeros))

    def forward_pass(self, x):
        return tf.stack([self.loc_net_list[i](x) for i in range(self.batch_size)])

    def forward_pass_test(self, x): # without dropout
        return tf.stack([self.loc_net_list[i].test(x) for i in range(self.batch_size)])

    def forward_pass_batch(self, x):
        return tf.stack([self.loc_net_list[i](x[i]) for i in range(self.batch_size)])



class BayesianDensityNetwork(tf.keras.Model):
    """Multilayer fully-connected Bayesian neural network, with
    two heads to predict both the mean and the standard deviation.

    Parameters
    ----------
    units : List[int]
        Number of output dimensions for each layer
        in the core network.
    units : List[int]
        Number of output dimensions for each layer
        in the head networks.
    name : None or str
        Name for the layer
    """

    def __init__(self, units, head_units, name=None):
        # Initialize
        super(BayesianDensityNetwork, self).__init__(name=name)

        # Create sub-networks
        self.core_net = BayesianDenseNetwork(units)
        self.d_out = head_units[-1]
        self.loc_net = BayesianDenseNetwork([units[-1]] + head_units)
        self.std_net = BayesianDenseNetwork([units[-1]] + head_units)

    def call(self, x, sampling=True):
        """Pass data through the model

        Parameters
        ----------
        x : tf.Tensor
            Input data
        sampling : bool
            Whether to sample parameter values from their
            variational distributions (if True, the default), or
            just use the Maximum a Posteriori parameter value
            estimates (if False).

        Returns
        -------
        preds : tf.Tensor of shape (Nsamples, 2)
            Output of this model, the predictions.  First column is
            the mean predictions, and second column is the standard
            deviation predictions.
        """

        # Pass data through core network
        x = self.core_net(x, sampling=sampling)
        x = tf.nn.tanh(x)

        # Make predictions with each head network
        loc_preds = self.loc_net(x, sampling=sampling)
        std_preds = self.std_net(x, sampling=sampling)
        std_preds = tf.nn.softplus(std_preds)

        # Return mean and std predictions
        return tf.concat([loc_preds, std_preds], 1)

    def log_likelihood(self, x, y, sampling=True):
        """Compute the log likelihood of y given x"""

        # Compute mean and std predictions
        preds = self.call(x, sampling=sampling)

        # Return log likelihood of true data given predictions
        return ds.Normal(preds[:, :self.d_out], preds[:, -self.d_out:]).log_prob(y)

    @tf.function
    def sample(self, x):
        """Draw one sample from the predictive distribution"""
        preds = self.call(x)
        return ds.Normal(preds[:, :self.d_out], preds[:, -self.d_out:]).sample()

    def samples(self, x, n_samples=1):
        """Draw multiple samples from predictive distributions"""
        samples = np.zeros((n_samples, x.shape[0], self.d_out ))
        for i in range(n_samples):
            samples[i] = self.sample(x)
        return samples

    @property
    def losses(self):
        """Sum of the KL divergences between priors + posteriors"""
        return (self.core_net.losses +
                self.loc_net.losses +
                self.std_net.losses)


class BayesianDenseRegression(tf.keras.Model):
    """A multilayer Bayesian neural network regression

    Parameters
    ----------
    dims : List[int]
        List of units in each layer
    name : str
        Name for the network

    Attributes
    ----------
    losses : tensorflow.Tensor
        Sum of the Kullback–Leibler divergences between
        the posterior distributions and their priors,
        over all layers in the network

    Methods
    -------
    call : tensorflow.Tensor
        Perform the forward pass of the data through
        the network, predicting both means and stds
    log_likelihood : tensorflow.Tensor
        Compute the log likelihood of y given x
    samples : tensorflow.Tensor
        Draw multiple samples from the predictive distribution
    """

    def __init__(self, dims, name=None):

        super(BayesianDenseRegression, self).__init__(name=name)

        # Multilayer fully-connected neural network to predict mean
        self.loc_net = BayesianDenseNetwork(dims)
        self.d_out = self.loc_net.d_out
        # Variational distribution variables for observation error
        self.std_alpha = tf.Variable([10.0]*self.d_out, name='std_alpha')
        self.std_beta = tf.Variable([10.0]*self.d_out, name='std_beta')

    def forward_pass(self, x, sampling=True):
        return self.loc_net(x, sampling=sampling)

    def call(self, x, sampling=True):
        """Perform forward pass, predicting both means + stds"""

        # Predict means
        loc_preds = self.forward_pass(x, sampling=True)

        # Predict std deviation
        posterior = ds.Gamma(self.std_alpha, self.std_beta)
        transform = lambda x: tf.sqrt(tf.math.reciprocal(x))
        N = x.shape[0]
        if sampling:
            std_preds = transform(posterior.sample([N]))
        else:
            std_preds = tf.ones([N, 1]) * transform(posterior.mean())
        # Return mean and std predictions
        return tf.concat([loc_preds, std_preds], 1)

    def log_likelihood(self, x, y, sampling=True):
        """Compute the log likelihood of y given x"""

        # Compute mean and std predictions
        preds = self.call(x, sampling=sampling)

        # Return log likelihood of true data given predictions
        return ds.Normal(preds[:, :self.d_out], preds[:, -self.d_out:]).log_prob(y)

    @tf.function
    def sample(self, x):
        """Draw one sample from the predictive distribution"""
        preds = self.call(x)
        return ds.Normal(preds[:, :self.d_out], preds[:, -self.d_out:]).sample()

    def samples(self, x, n_samples=1):
        """Draw multiple samples from the predictive distribution"""
        samples = np.zeros((n_samples, x.shape[0], self.d_out))
        for i in range(n_samples):
            samples[i] = self.sample(x)
        return samples

    @property
    def losses(self):
        """Sum of the KL divergences between priors + posteriors"""

        # Loss due to network weights
        net_loss = self.loc_net.losses

        # Loss due to std deviation parameter
        posterior = ds.Gamma(self.std_alpha, self.std_beta)
        prior = ds.Gamma([1.0]*self.d_out, [1.0]*self.d_out)
        std_loss = tf.reduce_mean(ds.kl_divergence(posterior, prior))
        # Return the sum of both
        return net_loss + std_loss


